/*
 * emager_app.c -- see emager_app.h.
 *
 * Two implementation notes worth reading before changing anything here.
 *
 * 1. The MAV is a RUNNING sum, not a re-scan.
 *    A 200-sample window advancing by 10 would cost 200*64 = 12800 absolute values per
 *    prediction if recomputed from scratch, 200 times per second. Keeping a per-channel
 *    sum and adding the incoming / subtracting the outgoing sample makes it 128
 *    operations instead of 12800, and it is exact -- these are integers, so there is no
 *    drift to worry about, which is the usual objection to running sums.
 *
 * 2. The smoothing is a majority vote over a tiny ring, not a mean.
 *    Averaging class indices is meaningless (the classes are not ordered), and
 *    averaging logits would need the score array kept per window for no real benefit.
 */

#include "emager_app.h"

#include <string.h>

#include "emager_nn_port.h"
#include "emager_prototypes.h"   /* EMAGER_FACTORY_PROTOTYPES (generated) */

/* --- state --------------------------------------------------------------- */

/* Sliding window of |x|. uint16 is enough: samples are int16, so |x| <= 32768.
 * 200 * 64 * 2 = 25.6 KB -- negligible on an STM32N6, but the single biggest RAM
 * consumer here, so it is the first thing to shrink on a smaller part (a shorter
 * window would change the features and is NOT the way to do it; storing int8 deltas
 * would be). */
static uint16_t s_abs_window[EMAGER_WINDOW_SIZE][EMAGER_N_FEATURES];
static uint32_t s_abs_sum[EMAGER_N_FEATURES];
static uint16_t s_write_idx;        /* next slot to overwrite                     */
static uint16_t s_filled;           /* samples seen, saturating at WINDOW_SIZE    */
static uint16_t s_since_last;       /* samples since the last emitted window      */
static volatile int s_window_ready;

static float   s_mav[EMAGER_N_FEATURES];
static int8_t  s_q_in[EMAGER_N_FEATURES];
static int8_t  s_q_embed[EMAGER_EMBED_DIM];
static float   s_embed[EMAGER_EMBED_DIM];

static emager_prototypes_t s_protos;
static emager_calib_t      s_calib;
static uint8_t             s_calib_class;
static emager_app_mode_t   s_mode = EMAGER_MODE_IDLE;

static int8_t  s_history[EMAGER_SMOOTHING_WINDOWS];
static uint8_t s_history_idx;
static uint8_t s_history_filled;
static int     s_prediction     = -1;
static int     s_raw_prediction = -1;

/* Weak so a firmware that never defines it still links. */
#if defined(__GNUC__)
__attribute__((weak))
#endif
void emager_app_log(const char *msg)
{
    (void)msg;
}

/* --- init ---------------------------------------------------------------- */

int emager_app_init(const emager_prototypes_t *stored)
{
    memset(s_abs_window, 0, sizeof(s_abs_window));
    memset(s_abs_sum, 0, sizeof(s_abs_sum));
    s_write_idx = 0;
    s_filled = 0;
    s_since_last = 0;
    s_window_ready = 0;
    s_prediction = -1;
    s_raw_prediction = -1;
    s_history_idx = 0;
    s_history_filled = 0;
    s_mode = EMAGER_MODE_IDLE;

    if (stored != NULL) {
        /* The caller is responsible for having checked that this blob belongs to the
         * CURRENT model. Prototypes paired with different weights fail silently -- as
         * wrong predictions, never as an error -- so that check cannot be skipped and
         * cannot be done here (only the caller knows the storage format). */
        memcpy(&s_protos, stored, sizeof(s_protos));
    } else {
        memcpy(&s_protos, &EMAGER_FACTORY_PROTOTYPES, sizeof(s_protos));
    }

    if (emager_nn_init() != 0) {
        emager_app_log("emager: network init failed");
        return -1;
    }

    s_mode = EMAGER_MODE_PREDICT;
    return 0;
}

/* --- acquisition --------------------------------------------------------- */

int emager_app_push_sample(const int16_t channels[EMAGER_N_FEATURES])
{
    for (int c = 0; c < EMAGER_N_FEATURES; ++c) {
        int32_t v = channels[c];
        uint16_t a = (uint16_t)((v < 0) ? -v : v);
        /* Subtract the sample leaving the window before adding the new one; while the
         * window is still filling, the slot holds 0 so this is a no-op. */
        s_abs_sum[c] -= s_abs_window[s_write_idx][c];
        s_abs_sum[c] += a;
        s_abs_window[s_write_idx][c] = a;
    }

    s_write_idx = (uint16_t)((s_write_idx + 1u) % EMAGER_WINDOW_SIZE);
    if (s_filled < EMAGER_WINDOW_SIZE) {
        s_filled++;
    }
    if (s_since_last < 0xFFFFu) {
        s_since_last++;
    }

    /* Only emit once the window is genuinely full: a partial window would produce a MAV
     * scaled by the wrong divisor and the first predictions after boot would be
     * confidently wrong rather than absent. */
    if (s_filled >= EMAGER_WINDOW_SIZE && s_since_last >= EMAGER_WINDOW_INCREMENT) {
        s_since_last = 0;
        s_window_ready = 1;
        return 1;
    }
    return 0;
}

int emager_app_window_ready(void)
{
    return s_window_ready;
}

/* --- inference ----------------------------------------------------------- */

static int smooth(int raw)
{
    if (EMAGER_SMOOTHING_WINDOWS <= 1 || raw < 0) {
        return raw;
    }
    s_history[s_history_idx] = (int8_t)raw;
    s_history_idx = (uint8_t)((s_history_idx + 1u) % EMAGER_SMOOTHING_WINDOWS);
    if (s_history_filled < EMAGER_SMOOTHING_WINDOWS) {
        s_history_filled++;
    }

    uint8_t counts[EMAGER_N_CLASSES];
    memset(counts, 0, sizeof(counts));
    for (uint8_t i = 0; i < s_history_filled; ++i) {
        int8_t c = s_history[i];
        if (c >= 0 && c < EMAGER_N_CLASSES) {
            counts[c]++;
        }
    }

    int best = raw;
    uint8_t best_n = 0;
    for (int c = 0; c < EMAGER_N_CLASSES; ++c) {
        if (counts[c] > best_n) {
            best_n = counts[c];
            best = c;
        }
    }
    return best;
}

int emager_app_process(void)
{
    if (!s_window_ready) {
        return -1;
    }
    s_window_ready = 0;

    /* MAV = mean(|x|) over the window, exactly as training computes it. */
    for (int c = 0; c < EMAGER_N_FEATURES; ++c) {
        s_mav[c] = (float)s_abs_sum[c] / (float)EMAGER_WINDOW_SIZE;
    }

    emager_quantize_input(s_mav, s_q_in);
    if (emager_nn_forward(s_q_in, s_q_embed) != 0) {
        emager_app_log("emager: forward failed");
        return -1;
    }
    emager_dequantize_embedding(s_q_embed, s_embed);

    if (s_mode == EMAGER_MODE_CALIBRATE) {
        /* Accumulate, do not classify: during calibration the prototypes are precisely
         * what is not trustworthy yet. */
        emager_calib_add(&s_calib, s_embed, s_calib_class);
        return -1;
    }

    s_raw_prediction = emager_classify(&s_protos, s_embed, NULL);
    s_prediction = smooth(s_raw_prediction);
    return s_prediction;
}

int emager_app_get_prediction(void)      { return s_prediction; }
int emager_app_get_raw_prediction(void)  { return s_raw_prediction; }
const float *emager_app_get_mav(void)    { return s_mav; }
emager_app_mode_t emager_app_get_mode(void) { return s_mode; }
const emager_prototypes_t *emager_app_get_prototypes(void) { return &s_protos; }

/* --- calibration --------------------------------------------------------- */

void emager_app_calib_start(void)
{
    emager_calib_reset(&s_calib);
    s_calib_class = 0;
    s_mode = EMAGER_MODE_CALIBRATE;
    /* Drop the smoothing history: votes cast against the old prototypes mean nothing
     * once the new ones are in. */
    s_history_idx = 0;
    s_history_filled = 0;
}

void emager_app_calib_set_class(uint8_t cls)
{
    if (cls < EMAGER_N_CLASSES) {
        s_calib_class = cls;
    }
}

int emager_app_calib_finish(emager_prototypes_t *out)
{
    if (out == NULL) {
        return 0;
    }
    /* Finalize into the caller's scratch struct, never straight into s_protos: an
     * abandoned or incomplete calibration must leave the device exactly as it was, not
     * half-configured. */
    int n_ok = emager_calib_finalize(&s_calib, out);
    s_mode = EMAGER_MODE_PREDICT;

    if (n_ok != EMAGER_N_CLASSES) {
        emager_app_log("emager: calibration incomplete, keeping previous prototypes");
        return 0;
    }
    memcpy(&s_protos, out, sizeof(s_protos));
    s_prediction = -1;
    s_raw_prediction = -1;
    return 1;
}

uint32_t emager_app_calib_count(uint8_t cls)
{
    if (cls >= EMAGER_N_CLASSES) {
        return 0;
    }
    return s_calib.n[cls];
}
