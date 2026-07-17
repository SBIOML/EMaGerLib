/*
 * emager_proto.c -- see emager_proto.h for the design and the calibration flow.
 *
 * Plain C99. No allocation, no float64, no libc beyond <string.h> and <math.h>.
 * Every loop bound is a compile-time constant from emager_model_params.h, so this
 * unrolls/vectorizes fine on Cortex-M and costs a few microseconds.
 */

#include "emager_proto.h"

#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 *  Model I/O quantization
 *
 *  TFLite affine quantization:  real = scale * (q - zero_point)
 *  so the inverse is:           q    = real / scale + zero_point
 * ------------------------------------------------------------------------- */

void emager_quantize_input(const float mav[EMAGER_N_FEATURES],
                           int8_t q_out[EMAGER_N_FEATURES])
{
    for (int i = 0; i < EMAGER_N_FEATURES; ++i) {
        float q = roundf(mav[i] / EMAGER_INPUT_SCALE) + (float)EMAGER_INPUT_ZERO_POINT;
        /* Saturate. A MAV spike must clip to the end of the range, not wrap around
         * to the opposite sign, which is what a bare (int8_t) cast would do. */
        if (q < -128.0f) q = -128.0f;
        if (q >  127.0f) q =  127.0f;
        q_out[i] = (int8_t)q;
    }
}

void emager_dequantize_embedding(const int8_t q_embed[EMAGER_EMBED_DIM],
                                 float embed_out[EMAGER_EMBED_DIM])
{
    for (int i = 0; i < EMAGER_EMBED_DIM; ++i) {
        embed_out[i] = ((float)q_embed[i] - (float)EMAGER_EMBED_ZERO_POINT)
                       * EMAGER_EMBED_SCALE;
    }
}

/* -------------------------------------------------------------------------
 *  Inference
 * ------------------------------------------------------------------------- */

int emager_classify(const emager_prototypes_t *protos,
                    const float embed[EMAGER_EMBED_DIM],
                    float scores[EMAGER_N_CLASSES])
{
    int   best       = -1;
    float best_score = 0.0f;

    for (int c = 0; c < EMAGER_N_CLASSES; ++c) {
        if (!protos->valid[c]) {
            /* Never a candidate: an uncalibrated class has no meaningful location in
             * embedding space (see the note in emager_proto.h). */
            if (scores) {
                scores[c] = -INFINITY;
            }
            continue;
        }

        float dist = 0.0f;
        for (int i = 0; i < EMAGER_EMBED_DIM; ++i) {
            const float diff = embed[i] - protos->v[c][i];
            dist += diff * diff;
        }

        /* Matches the PyTorch model's logit exactly: -||e - p_c||^2 / D.
         * The /D does not change the argmax, but it keeps a softmax over `scores`
         * numerically equal to predict_proba() on the host. */
        const float score = -dist / (float)EMAGER_EMBED_DIM;

        if (scores) {
            scores[c] = score;
        }
        if (best < 0 || score > best_score) {
            best       = c;
            best_score = score;
        }
    }

    return best;
}

/* -------------------------------------------------------------------------
 *  On-device calibration
 * ------------------------------------------------------------------------- */

void emager_calib_reset(emager_calib_t *c)
{
    memset(c, 0, sizeof(*c));
}

void emager_calib_add(emager_calib_t *c,
                      const float embed[EMAGER_EMBED_DIM],
                      uint8_t cls)
{
    if (cls >= EMAGER_N_CLASSES) {
        return;  /* ignore rather than corrupt neighbouring prototypes */
    }
    for (int i = 0; i < EMAGER_EMBED_DIM; ++i) {
        c->sum[cls][i] += embed[i];
    }
    c->n[cls] += 1u;
}

int emager_calib_finalize(const emager_calib_t *c, emager_prototypes_t *out)
{
    int n_valid = 0;

    for (int cls = 0; cls < EMAGER_N_CLASSES; ++cls) {
        if (c->n[cls] == 0u) {
            /* Mark invalid AND zero it, so a stale prototype from a previous
             * calibration cannot survive in a slot the user just skipped. */
            memset(out->v[cls], 0, sizeof(out->v[cls]));
            out->valid[cls] = 0u;
            continue;
        }

        const float inv_n = 1.0f / (float)c->n[cls];
        for (int i = 0; i < EMAGER_EMBED_DIM; ++i) {
            out->v[cls][i] = c->sum[cls][i] * inv_n;
        }
        out->valid[cls] = 1u;
        ++n_valid;
    }

    return n_valid;
}
