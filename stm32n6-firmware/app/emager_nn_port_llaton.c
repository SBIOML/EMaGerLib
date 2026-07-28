/*
 * emager_nn_port_llaton.c -- emager_nn_port.h on the STM32N6 Neural-ART NPU.
 *
 * READ THIS BEFORE USING THE FILE
 * -------------------------------
 * The three TODO blocks below are NOT filled in, and that is deliberate.
 *
 * The exact LL_ATON entry points, the name of the generated header, and the way input
 * and output buffers are handed to the network all depend on your X-CUBE-AI version and
 * on the --name you passed to `stedgeai generate`. Writing plausible-looking calls here
 * would produce a file that compiles in nobody's project and sends you debugging an
 * invented API instead of reading the one you actually have. A TODO you fill in from
 * the generated header takes ten minutes; a wrong API costs an afternoon.
 *
 * What to look for, in order:
 *
 *   1. In model/generated/ (or wherever `stedgeai generate --output` wrote):
 *        <name>.h                  -- the generated network's interface
 *        <name>_generate_report.txt-- input/output shapes, dtypes, memory pools
 *        ll_aton_rt_user_api.h     -- the runtime's user-facing API
 *
 *   2. The calls you need are, by role:
 *        LL_ATON_RT_Init_Network / LL_ATON_RT_DeInit_Network
 *        LL_ATON_RT_Main  (or the RunEpochBlock loop, depending on version)
 *        LL_ATON_Set_User_Input_Buffer_Default  / LL_ATON_Set_User_Output_Buffer_Default
 *            -- only needed when you want to supply your own buffers rather than use
 *               the ones the generated code allocates.
 *
 *   3. A working reference: ST's STM32N6 "getting started" application packages on
 *      https://github.com/STMicroelectronics. Read their inference loop; ignore their
 *      camera and image preprocessing, which is all that differs from this use case.
 *
 * Two traps specific to this model:
 *
 *   - CACHE. The N6 has caches in front of the memory the NPU reads and writes. If the
 *     CPU writes the input buffer and the NPU reads stale memory (or the CPU reads a
 *     stale output), you get plausible but wrong embeddings -- exactly the failure the
 *     self-test reports as "embeddings differ" and exactly the one people blame on the
 *     model. Clean before the run, invalidate after, or place the buffers in
 *     non-cacheable memory via the MPU. This is the single most common N6 porting bug.
 *
 *   - LAYOUT. The generated network may expect NHWC and a specific tensor rank. This
 *     model's input is a flat [1, 64] MAV vector, so there is little room for error --
 *     but check the shape in the generate report rather than assuming.
 *
 * Until this file is finished, build with emager_nn_port_stub.c instead: the
 * application layer, the windowing and the calibration flow can all be exercised
 * without a working NPU.
 */

#include "emager_nn_port.h"

#include <string.h>

/* TODO(1): include the generated network header, e.g.
 *     #include "emager.h"
 *     #include "ll_aton_runtime.h"
 */

/* Buffers are static and file-scope so nothing here allocates and the addresses stay
 * fixed -- convenient when they have to be registered with the runtime once, and
 * necessary if you end up placing them in a specific memory region.
 *
 * If you need them in a particular section (non-cacheable, or a pool the generate
 * report names), attach the attribute here rather than scattering it through the app:
 *     __attribute__((section(".psram_bss"), aligned(32)))
 */
static int8_t s_in_buf[EMAGER_N_FEATURES];
static int8_t s_out_buf[EMAGER_EMBED_DIM];

static int s_ready;

int emager_nn_init(void)
{
    s_ready = 0;

    /* TODO(2): initialise the network and, if you are supplying your own buffers,
     * register s_in_buf / s_out_buf with it. Something along the lines of:
     *
     *     LL_ATON_RT_RuntimeInit();
     *     LL_ATON_RT_Init_Network(&NN_Instance_emager);
     *     LL_ATON_Set_User_Input_Buffer_Default(0, s_in_buf, sizeof(s_in_buf));
     *     LL_ATON_Set_User_Output_Buffer_Default(0, s_out_buf, sizeof(s_out_buf));
     *
     * Check the return codes -- every one of them. A network that failed to initialise
     * still "runs" and returns whatever is in the output buffer, which on a fresh boot
     * is zeros: a perfectly stable, perfectly wrong embedding.
     */

    /* Sanity-check the dimensions the firmware was compiled against against the ones
     * the generated network reports. They come from two different files (the .tflite's
     * emager_model_params.h and the generated network) and nothing else forces them to
     * agree -- re-exporting the model without regenerating the network, or the reverse,
     * is easy and silently produces garbage.
     *
     * TODO(3): replace with the generated accessors, e.g. comparing against the shape
     * in <name>.h, then delete this comment.
     */

    s_ready = 1;
    return 0;
}

int emager_nn_forward(const int8_t in[EMAGER_N_FEATURES],
                      int8_t out[EMAGER_EMBED_DIM])
{
    if (!s_ready) {
        return -1;
    }

    memcpy(s_in_buf, in, sizeof(s_in_buf));

    /* TODO(2, continued): clean the cache over s_in_buf, run the network, invalidate
     * the cache over s_out_buf. Roughly:
     *
     *     SCB_CleanDCache_by_Addr((uint32_t *)s_in_buf, sizeof(s_in_buf));
     *     LL_ATON_RT_Main(&NN_Instance_emager);
     *     SCB_InvalidateDCache_by_Addr((uint32_t *)s_out_buf, sizeof(s_out_buf));
     *
     * and return non-zero if the run reports an error.
     */

    memcpy(out, s_out_buf, sizeof(s_out_buf));
    return 0;
}
