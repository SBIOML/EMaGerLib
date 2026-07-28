/*
 * emager_nn_port_stub.c -- a fake network, for bringing the rest up first.
 *
 * Compile this INSTEAD OF emager_nn_port_llaton.c to exercise everything that is not
 * the NPU: the sliding window, the MAV, the calibration state machine, the UI, the
 * serial trace. All of that is ordinary C and none of it needs a working Neural-ART
 * port -- but debugging it at the same time as the NPU is how a two-hour problem
 * becomes a two-day one.
 *
 * The embedding it produces is deterministic nonsense: a cheap hash of the input. That
 * is enough for calibration to converge on something self-consistent (the same gesture
 * gives the same embedding, so prototypes form and classification is stable), which is
 * exactly what you need to test the flow.
 *
 * It will NOT pass emager_selftest_run(), and it must not: the self-test compares
 * against embeddings the real model produced. A stub that passed it would be a stub
 * that had made the self-test worthless.
 */

#include "emager_nn_port.h"

int emager_nn_init(void)
{
    return 0;
}

int emager_nn_forward(const int8_t in[EMAGER_N_FEATURES],
                      int8_t out[EMAGER_EMBED_DIM])
{
    /* Deterministic, input-dependent, cheap. Each output element mixes a different
     * strided slice of the input so that different gestures land in different places
     * rather than everything collapsing onto one point. */
    for (int j = 0; j < EMAGER_EMBED_DIM; ++j) {
        int32_t acc = (int32_t)(j * 31);
        for (int i = j % 4; i < EMAGER_N_FEATURES; i += 4) {
            acc += (int32_t)in[i] * (int32_t)((i + j) % 7 + 1);
        }
        out[j] = (int8_t)(acc % 127);
    }
    return 0;
}
