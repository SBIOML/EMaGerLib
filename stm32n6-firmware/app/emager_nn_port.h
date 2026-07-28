/*
 * emager_nn_port.h -- the only thing the application knows about the neural network.
 *
 * Two functions. Everything above this line (windowing, MAV, classification,
 * calibration) is portable C that can be compiled and tested on a PC; everything below
 * it is target- and runtime-specific.
 *
 * The split is not decoration. TFLite Micro, X-CUBE-AI's ai_* API and the STM32N6
 * Neural-ART (LL_ATON) runtime have entirely different entry points, initialisation
 * and buffer conventions. Letting any of that leak into the application would mean
 * rewriting the application to change board -- and, more immediately, would make it
 * impossible to run the self-test against a stub while the real port is still being
 * wired up.
 *
 * Implementations live in emager_nn_port_<runtime>.c. Exactly one gets compiled.
 */

#ifndef EMAGER_NN_PORT_H
#define EMAGER_NN_PORT_H

#include <stdint.h>

#include "emager_model_params.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bring the network up: allocate/assign buffers, initialise the runtime, check that
 * the model the firmware was built against is the one actually present.
 *
 * Returns 0 on success. Call once, before any forward pass.
 */
int emager_nn_init(void);

/*
 * One inference: quantized features in, quantized embedding out.
 *
 * INT8 on both sides deliberately. The scale/zero-point conversions belong to
 * emager_proto.c, which reads them from constants generated off the real .tflite --
 * duplicating them in the port is how a firmware ends up quantizing with values that
 * no longer match the flashed model.
 *
 * Returns 0 on success, non-zero on an inference error. Must be safe to call
 * repeatedly; must not allocate.
 */
int emager_nn_forward(const int8_t in[EMAGER_N_FEATURES],
                      int8_t out[EMAGER_EMBED_DIM]);

#ifdef __cplusplus
}
#endif

#endif /* EMAGER_NN_PORT_H */
