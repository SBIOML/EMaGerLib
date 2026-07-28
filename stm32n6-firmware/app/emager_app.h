/*
 * emager_app.h -- the EMaGer application layer: raw samples in, gesture out.
 *
 * What this owns
 * --------------
 *   raw EMG samples (4x16 grid)
 *        |  emager_app_push_sample()   -- sliding window, running |x| sum
 *   MAV[64] every EMAGER_WINDOW_INCREMENT samples
 *        |  emager_quantize_input() + emager_nn_forward()
 *   INT8 embedding
 *        |  emager_dequantize_embedding() + emager_classify()
 *   gesture index  ->  emager_app_get_prediction()
 *
 * It does NOT own acquisition (that is your ADC/SPI driver) nor inference (that is
 * emager_nn_port.h). It owns the part in between, which is where ports actually go
 * wrong: the window has to be the one the model was trained on, to the sample.
 *
 * The MAV parameters below MUST match training. They are not tunable knobs -- change
 * one and the flashed model is being fed features it has never seen, which shows up as
 * plausible-looking but wrong predictions rather than as an error.
 *
 * Plain C99. No allocation, no RTOS dependency. Everything is statically sized.
 */

#ifndef EMAGER_APP_H
#define EMAGER_APP_H

#include <stdint.h>

#include "emager_model_params.h"
#include "emager_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --- fixed by training; see examples/deployment/export_model.py ----------- */
#define EMAGER_WINDOW_SIZE       200   /* samples per window                  */
#define EMAGER_WINDOW_INCREMENT  10    /* samples between two predictions     */
#define EMAGER_SAMPLING_HZ       2000  /* -> a prediction every 5 ms          */

/* Majority vote over the last N predictions. Raw per-window output is jittery at
 * gesture transitions; a prosthetic hand that twitches on every ambiguous window is
 * unusable. 5 windows = 25 ms of lag, which is imperceptible. Set to 1 to disable. */
#define EMAGER_SMOOTHING_WINDOWS 5

typedef enum {
    EMAGER_MODE_IDLE = 0,   /* not predicting                                    */
    EMAGER_MODE_PREDICT,    /* normal operation                                  */
    EMAGER_MODE_CALIBRATE,  /* feeding windows into a prototype being rebuilt    */
} emager_app_mode_t;

/*
 * Bring the application up: network, prototypes, smoothing state.
 *
 * `stored` is a prototype set read back from your persistent storage, or NULL to start
 * from the factory defaults. Pass NULL until persistence exists -- the factory
 * prototypes work uncalibrated, that is what they are for.
 *
 * Returns 0 on success.
 */
int emager_app_init(const emager_prototypes_t *stored);

/*
 * Feed one sample of all 64 channels, in the SAME channel order as training.
 *
 * Call this from your acquisition path at EMAGER_SAMPLING_HZ. It is O(64) -- a running
 * sum, not a re-scan of the window -- so it is safe from an interrupt if your
 * inference is not; see emager_app_window_ready().
 *
 * Returns 1 when this sample completed a window (a new MAV vector is available),
 * 0 otherwise.
 */
int emager_app_push_sample(const int16_t channels[EMAGER_N_FEATURES]);

/*
 * True when a window is ready to be processed.
 *
 * Split from push_sample() so acquisition can stay in an ISR while the forward pass
 * runs in the main loop. Running an NPU inference inside a 2 kHz interrupt is not a
 * good idea.
 */
int emager_app_window_ready(void);

/*
 * Consume the pending window: MAV -> quantize -> forward -> classify -> smooth.
 *
 * Returns the smoothed gesture index, or -1 if there was nothing to do, the inference
 * failed, or no prototype is valid. Clears the ready flag.
 */
int emager_app_process(void);

/* Last smoothed prediction, or -1 if there is not one yet. */
int emager_app_get_prediction(void);

/* Raw (unsmoothed) prediction of the most recent window, or -1. Useful when
 * diagnosing whether jitter comes from the model or from the smoothing. */
int emager_app_get_raw_prediction(void);

/* The most recent MAV vector, for debugging or for streaming to a host. */
const float *emager_app_get_mav(void);

/* --- on-device calibration ------------------------------------------------
 *
 * The whole reason the prototypical model was chosen: adapting to a user needs a
 * forward pass and an average, no gradients and no training data kept.
 *
 * Flow:
 *     emager_app_calib_start();
 *     for each gesture c:  emager_app_calib_set_class(c);
 *                          ... user holds the gesture, keep calling push_sample/process
 *     emager_app_calib_finish(&fresh);   -> persist `fresh` if it returns 1
 */

/* Enter calibration mode and discard any previous partial calibration. */
void emager_app_calib_start(void);

/* Which gesture the user is performing now. Windows are accumulated into it until
 * this is called again. */
void emager_app_calib_set_class(uint8_t cls);

/*
 * Finish calibration, writing the new prototypes to `out`.
 *
 * Returns 1 if every class got at least one window, 0 otherwise -- in which case the
 * live prototypes are left untouched and `out` must not be used. A class with no
 * samples is invalid, not zero: a zeroed prototype sits at the origin of embedding
 * space and would attract anything near it, so a user who skipped a gesture would get
 * a class that fires at random.
 *
 * On success the live prototypes are replaced. Persisting `out` is the caller's job.
 */
int emager_app_calib_finish(emager_prototypes_t *out);

/* Number of windows collected so far for `cls` -- drive a progress bar with it. */
uint32_t emager_app_calib_count(uint8_t cls);

emager_app_mode_t emager_app_get_mode(void);

/* The live prototypes, e.g. to persist them or to hash them against the model. */
const emager_prototypes_t *emager_app_get_prototypes(void);

/*
 * Optional trace hook. Left unimplemented on purpose -- define it in your firmware to
 * route messages to the ST-LINK VCP UART (SWO does not work on the NUCLEO-N657X0-Q).
 * The default is a weak no-op, so nothing breaks if you never define it.
 */
void emager_app_log(const char *msg);

#ifdef __cplusplus
}
#endif

#endif /* EMAGER_APP_H */
