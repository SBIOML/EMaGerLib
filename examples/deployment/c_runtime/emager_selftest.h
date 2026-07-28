/*
 * emager_selftest.h -- on-device verification of a ported EMaGer model.
 *
 * What this answers
 * -----------------
 * "Does the model on this board compute the same thing as the model on the host?"
 *
 * export_model.py checks the exported artifacts against PyTorch, but it runs on a PC.
 * It cannot see your firmware: the MAV features you feed in, the input quantization,
 * the kernels your runtime chose. Those are where ports actually break, and they only
 * exist on the target.
 *
 * `make_test_vectors.py` emits emager_test_vectors.h -- real held-out windows plus the
 * INT8 embedding the host's model produced for each. This file replays them through
 * YOUR forward pass and reports where the two disagree.
 *
 * Reading the result
 * ------------------
 * The embedding is checked before the class, deliberately. The two failures have
 * completely different causes and confusing them wastes days:
 *
 *   embeddings match, class differs   -> the network is fine. The prototypes are the
 *                                        problem: a stale persisted blob, wrong `valid`
 *                                        flags, or simply a device that has been
 *                                        calibrated (which legitimately changes the
 *                                        class for the same embedding -- see below).
 *
 *   embeddings differ                 -> the network path. In order of likelihood:
 *                                        the MAV feature does not match training
 *                                        (window size, increment, channel order), the
 *                                        input quantization is wrong, or the runtime
 *                                        is mis-wired (tensor layout, missing op).
 *
 * IMPORTANT: the class column of the test vectors describes the FACTORY prototypes.
 * Run the self-test before on-device calibration, or against a fresh copy of
 * EMAGER_FACTORY_PROTOTYPES. After calibration a class mismatch is expected and means
 * nothing -- `embed_max_abs_err` is the number that stays meaningful.
 *
 * Plain C99, no allocation, no printf. Feed the result struct to whatever logging you
 * have (SWO, UART, an LED).
 */

#ifndef EMAGER_SELFTEST_H
#define EMAGER_SELFTEST_H

#include <stdint.h>

#include "emager_model_params.h"
#include "emager_proto.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Your firmware's forward pass: INT8 features in, INT8 embedding out.
 *
 * Taking a function pointer rather than calling the network directly is what keeps
 * this file independent of the inference runtime -- TFLite Micro, X-CUBE-AI's ai_*
 * API and the STM32N6 Neural-ART (LL_ATON) runtime all have different entry points,
 * and none of them belong in a verification routine.
 *
 * Return 0 on success, non-zero on an inference error.
 */
typedef int (*emager_nn_forward_fn)(const int8_t in[EMAGER_N_FEATURES],
                                    int8_t out[EMAGER_EMBED_DIM]);

typedef struct {
    uint16_t n_vectors;         /* how many were run                                  */
    uint16_t n_embed_ok;        /* embeddings within EMAGER_TEST_EMBED_TOL             */
    uint16_t n_class_ok;        /* classes matching the host (factory prototypes only) */
    uint16_t n_errors;          /* forward passes that returned non-zero               */
    int32_t  embed_max_abs_err; /* worst single-element deviation, in INT8 LSBs        */
    uint16_t worst_vector;      /* index of the vector that produced it                */
} emager_selftest_result_t;

/*
 * Replay every vector in emager_test_vectors.h.
 *
 * `protos` is the prototype set to classify against -- pass EMAGER_FACTORY_PROTOTYPES
 * (copied into a mutable struct) for the class column to be meaningful, or NULL to
 * check the network path only.
 *
 * Returns 1 if every vector passed both checks that were requested, 0 otherwise.
 * `res` is always filled in; inspect it either way.
 */
int emager_selftest_run(emager_nn_forward_fn forward,
                        const emager_prototypes_t *protos,
                        emager_selftest_result_t *res);

#ifdef __cplusplus
}
#endif

#endif /* EMAGER_SELFTEST_H */
