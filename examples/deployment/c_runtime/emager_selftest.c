/*
 * emager_selftest.c -- see emager_selftest.h.
 *
 * C99, no allocation, no libc beyond <string.h>. The only floating point is the
 * quantization of the stored MAV inputs, which is exactly what the real inference path
 * does -- the self-test must exercise emager_quantize_input(), not bypass it, because
 * a wrong input scale is one of the failures it exists to catch.
 */

#include "emager_selftest.h"

#include <string.h>

#include "emager_test_vectors.h"

int emager_selftest_run(emager_nn_forward_fn forward,
                        const emager_prototypes_t *protos,
                        emager_selftest_result_t *res)
{
    static int8_t q_in[EMAGER_N_FEATURES];
    static int8_t q_embed[EMAGER_EMBED_DIM];
    static float  embed[EMAGER_EMBED_DIM];

    if (res == NULL || forward == NULL) {
        return 0;
    }
    memset(res, 0, sizeof(*res));
    res->n_vectors = (uint16_t)EMAGER_TEST_N_VECTORS;

    for (uint16_t i = 0; i < (uint16_t)EMAGER_TEST_N_VECTORS; ++i) {
        /* Same entry point as emager_predict(): float MAV in, quantized here. */
        emager_quantize_input(EMAGER_TEST_MAV[i], q_in);

        if (forward(q_in, q_embed) != 0) {
            res->n_errors++;
            continue;
        }

        /* --- the network check -------------------------------------------
         * Compared in INT8, not after dequantization. The stored reference is the
         * raw tensor the host's interpreter produced, so comparing raw keeps the
         * tolerance in units anyone can reason about (LSBs) instead of units that
         * depend on the embedding scale.
         */
        int32_t worst = 0;
        for (int j = 0; j < EMAGER_EMBED_DIM; ++j) {
            int32_t d = (int32_t)q_embed[j] - (int32_t)EMAGER_TEST_EMBED[i][j];
            if (d < 0) { d = -d; }
            if (d > worst) { worst = d; }
        }
        if (worst > res->embed_max_abs_err) {
            res->embed_max_abs_err = worst;
            res->worst_vector = i;
        }
        if (worst <= EMAGER_TEST_EMBED_TOL) {
            res->n_embed_ok++;
        }

        /* --- the end-to-end check ----------------------------------------
         * Only meaningful against the factory prototypes; see the header.
         */
        if (protos != NULL) {
            emager_dequantize_embedding(q_embed, embed);
            int cls = emager_classify(protos, embed, NULL);
            if (cls >= 0 && (uint8_t)cls == EMAGER_TEST_CLASS[i]) {
                res->n_class_ok++;
            }
        }
    }

    if (res->n_errors != 0) {
        return 0;
    }
    if (res->n_embed_ok != res->n_vectors) {
        return 0;
    }
    if (protos != NULL && res->n_class_ok != res->n_vectors) {
        return 0;
    }
    return 1;
}
