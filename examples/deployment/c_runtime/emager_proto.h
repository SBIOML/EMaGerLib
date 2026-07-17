/*
 * emager_proto.h -- on-device prototype classification and calibration for the
 * EMaGer few-shot prototypical models.
 *
 * Why this file exists
 * --------------------
 * A prototypical model has no learned classifier. It classifies by nearest
 * prototype, where each prototype is the MEAN EMBEDDING of a few examples of a
 * gesture. Those examples come from the *user*, on the device, after flashing --
 * so the prototypes cannot be baked into the neural network at export time. They
 * live here instead, in a plain writable array.
 *
 * The split, and why it is here rather than in the .tflite:
 *
 *     MAV (64 float)
 *        |  emager_quantize_input()
 *     int8[64]
 *        |  TFLite Micro / CMSIS-NN   <- the network, frozen at flash time
 *     int8[D]
 *        |  emager_dequantize_embedding()
 *     float[D] embedding
 *        |  emager_classify()          <- this file, prototypes writable at runtime
 *     class index
 *
 * The network is the expensive part (~152k MAC for EmagerCNNRingStrided) and never
 * changes, so it belongs in an INT8 kernel. The prototype distance is C x D
 * multiply-adds (7 x 64 = 448 for the default model) -- utterly negligible -- and it
 * is the only part that must stay mutable. Keeping it in C is what makes on-device
 * calibration possible at all: no backprop, no optimizer, just a forward pass and a
 * per-class average.
 *
 * This mirrors the PyTorch model exactly. EmagerCNNProtoRingStridedPTQ brackets only
 * the embedding with QuantStub/DeQuantStub and computes the distance in FP32 -- the
 * same boundary drawn above.
 *
 * Generated companions (produced by examples/deployment/export_model.py):
 *     emager_model_params.h  dimensions + the INT8 scale/zero-point constants
 *     emager_prototypes.h    factory prototypes, as a compile-time default
 *     emager_model_data.h    the .tflite network as a C byte array
 *
 * Everything here is plain C99, no allocation, no libc beyond <string.h>/<math.h>.
 */

#ifndef EMAGER_PROTO_H
#define EMAGER_PROTO_H

#include <stdint.h>

#include "emager_model_params.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One prototype per gesture, plus a validity flag.
 *
 * `valid[c] == 0` means "class c has no prototype" and emager_classify() will never
 * return it. This matters: a zeroed prototype is NOT a neutral one -- it sits at the
 * origin of embedding space and would attract any sample near it. A user who skips a
 * gesture during calibration must not silently get a class that fires at random.
 *
 * Size: EMAGER_N_CLASSES * EMAGER_EMBED_DIM * 4 bytes (+ flags). For the default
 * 7 x 64 model that is 1792 bytes -- small enough to sit in RAM and be persisted to
 * flash/EEPROM after calibration. See EMAGER_PROTOTYPES_NBYTES below.
 */
typedef struct {
    float   v[EMAGER_N_CLASSES][EMAGER_EMBED_DIM];
    uint8_t valid[EMAGER_N_CLASSES];
} emager_prototypes_t;

#define EMAGER_PROTOTYPES_NBYTES ((uint32_t)sizeof(emager_prototypes_t))

/*
 * Running per-class accumulator used while the user records calibration gestures.
 * Kept separate from emager_prototypes_t so a failed/abandoned calibration cannot
 * corrupt the prototypes currently in use.
 */
typedef struct {
    float    sum[EMAGER_N_CLASSES][EMAGER_EMBED_DIM];
    uint32_t n[EMAGER_N_CLASSES];
} emager_calib_t;

/* -------------------------------------------------------------------------
 *  Model I/O quantization
 *  The scale/zero-point constants come from emager_model_params.h, which the
 *  export tool writes from the actual .tflite -- never hand-edit them.
 * ------------------------------------------------------------------------- */

/* MAV feature vector (float) -> the model's int8 input tensor. Saturates, does not wrap. */
void emager_quantize_input(const float mav[EMAGER_N_FEATURES],
                           int8_t q_out[EMAGER_N_FEATURES]);

/* The model's int8 output tensor -> float embedding, ready for emager_classify(). */
void emager_dequantize_embedding(const int8_t q_embed[EMAGER_EMBED_DIM],
                                 float embed_out[EMAGER_EMBED_DIM]);

/* -------------------------------------------------------------------------
 *  Inference
 * ------------------------------------------------------------------------- */

/*
 * Nearest-prototype classification.
 *
 * Returns the predicted class index, or -1 if no prototype is valid (i.e. the device
 * has never been calibrated and no factory defaults were loaded).
 *
 * `scores` may be NULL. When given, it receives the per-class score
 *     score[c] = -||embed - proto[c]||^2 / D
 * which is exactly the logit the PyTorch model produces, so a softmax over it matches
 * predict_proba(). Invalid classes get -INFINITY.
 */
int emager_classify(const emager_prototypes_t *protos,
                    const float embed[EMAGER_EMBED_DIM],
                    float scores[EMAGER_N_CLASSES]);

/* -------------------------------------------------------------------------
 *  On-device calibration ("few-shot")
 *
 *  Flow:
 *      emager_calib_reset(&cal);
 *      for each gesture c the user performs:
 *          for each recorded window:
 *              ... run the network -> embed ...
 *              emager_calib_add(&cal, embed, c);
 *      if (emager_calib_finalize(&cal, &protos) == EMAGER_N_CLASSES) { persist(&protos); }
 *
 *  No gradients, no optimizer, no training data retained -- just a per-class mean of
 *  embeddings. That is the entire "training" step this model needs on device.
 * ------------------------------------------------------------------------- */

void emager_calib_reset(emager_calib_t *c);

/* Accumulate one embedding for class `cls`. Out-of-range `cls` is ignored. */
void emager_calib_add(emager_calib_t *c,
                      const float embed[EMAGER_EMBED_DIM],
                      uint8_t cls);

/*
 * Turn the accumulator into prototypes. Returns the number of classes that got at
 * least one sample; classes with none are marked invalid rather than zeroed-and-used.
 * Compare the return against EMAGER_N_CLASSES to decide whether calibration was
 * complete enough to keep.
 *
 * `out` may safely be the prototypes currently in use -- it is fully overwritten.
 */
int emager_calib_finalize(const emager_calib_t *c, emager_prototypes_t *out);

#ifdef __cplusplus
}
#endif

#endif /* EMAGER_PROTO_H */
