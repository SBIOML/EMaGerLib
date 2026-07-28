"""
export_model.py -- Export a trained EMaGer CNN to microcontroller-ready formats.

Pipeline (each stage is numerically verified against the PyTorch reference):

    PyTorch (.pt)  ->  ONNX  ->  TFLite float32  ->  TFLite full-INT8  ->  C array

The input to every exported model is a single (1, 64) float32 MAV feature vector
(mean-absolute-value over a 200-sample window, 4x16 electrode grid = 64 channels),
matching the training pipeline in train_cnn_benchmark.py. Batch size is fixed to 1
so the artifacts drop straight onto a microcontroller (one inference at a time).

The artifacts cover all three deployment stacks:
    - STM32 X-CUBE-AI : feed model.onnx OR model_int8.tflite (it generates the C code)
    - TFLite Micro    : compile model_int8.tflite  (or the generated model_int8.h C array)
    - ONNX Runtime    : feed model.onnx

INT8 is a *full-integer* quantization (int8 input, int8 output, int8 weights),
calibrated on a representative set of real MAV windows -- this is what most
Cortex-M inference kernels (CMSIS-NN) require. The int8 input/output scale and
zero-point are printed and written to the report so you know how to feed the MCU.

Prototypical models: the graph stops at the embedding
-----------------------------------------------------
The few-shot prototypical variants classify by distance to prototypes, and prototypes
are USER DATA computed on the device after flashing. Folding them into the graph as
constants (`--proto-export baked`) would freeze them at flash time and make on-device
calibration impossible.

So by default (`--proto-export embedding`) the exported graph is the embedding
`f_theta` ALONE -- `[1,64] -> [1,D]` -- and the prototype distance ships as C source
over a writable array:

    MAV -> [ int8 TFLite embedding ] -> dequant -> [ C: distance to prototypes ] -> class
             frozen at flash time                    rewritable at runtime

This is the same boundary the PyTorch model already draws (the quantized proto variants
bracket only embed() with Quant/DeQuant stubs). It costs nothing: the network is ~152k
MAC, the distance is C*D = 448. Alongside the .tflite the tool emits emager_model_params.h,
emager_prototypes.h (factory defaults) and copies c_runtime/emager_proto.{h,c}
(classification + calibration). See examples/deployment/README.md.

Verification is end-to-end regardless of mode: the reference is always the PyTorch
model's FULL forward, and in embedding mode the tool re-attaches the prototype distance
to the artifact's output before comparing -- so what is checked is "does the device
predict the same class", not "is the embedding tensor close".

Usage
-----
    python examples/deployment/export_model.py                       # RingStrided, Test_EM_C7_R5
    python examples/deployment/export_model.py --model EmagerCNNRingStrided --dataset Test_EM_C7_R5
    python examples/deployment/export_model.py --model EmagerCNNProtoRingStrided   # + C runtime
    python examples/deployment/export_model.py --model EmagerCNNProtoRingStrided --proto-export baked
    python examples/deployment/export_model.py --checkpoint path/state_dict.pt --model EmagerCNNBase
    python examples/deployment/export_model.py --formats onnx tflite-int8 carray
    python examples/deployment/export_model.py --epochs 10 --calib-samples 300

Export the FP32 variant (EmagerCNNProtoRingStrided), not the torch.ao INT8 ones
(...PTQ/...QAT): those convert themselves during fit() and eager-mode quantized modules
have no ONNX lowering. This pipeline does its own INT8 via TFLiteConverter anyway, so
you get the same artifact. The tool refuses them with that message.

Requires: torch, onnx, onnxruntime, tensorflow, onnx2tf, tf_keras
(the "deploy" optional-dependency group -- `pip install -e ".[deploy]"`).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

# --- import order note (see CLAUDE.md "Windows torch/Qt DLL order") ----------
# This script touches torch (via the models / train helpers) and tensorflow but
# never a Qt binding, so the c10.dll WinError-1114 trap does not apply here.

# Quiet TensorFlow's C++ logging before it is imported anywhere downstream.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger(__name__)

# Make the training helpers importable (data pipeline + dataset info readers).
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = REPO_ROOT / "examples" / "training"
sys.path.insert(0, str(TRAINING_DIR))

# ----------------------------------------------------------------------------
#  Constants -- kept in lock-step with train_cnn_benchmark.py so exported models
#  see exactly the features they were trained on.
# ----------------------------------------------------------------------------

INPUT_SHAPE      = (4, 16)          # electrode grid; flattened to 64 MAV features
N_FEATURES       = int(np.prod(INPUT_SHAPE))
WINDOW_SIZE      = 200
WINDOW_INCREMENT = 10
SAMPLING         = 2000

DEFAULT_MODEL    = "EmagerCNNRingStrided"
DEFAULT_DATASET  = "Test_EM_C7_R5"


# ----------------------------------------------------------------------------
#  Data
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#  Model (train fresh, or load a cached state_dict)
# ----------------------------------------------------------------------------

def load_checkpoint_model(model_name, num_classes, checkpoint):
    """Instantiate the variant and load a cached state_dict. Returns an eval-mode
    CPU model whose forward takes (B, 64) and returns (B, num_classes)."""
    import torch
    import emagerlib.models.new_emager_cnn as new_models

    model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"  loaded checkpoint {checkpoint}  (missing={len(missing)} unexpected={len(unexpected)})")
    model.eval().cpu()
    return model


def train_model(model_name, num_classes, train_mav, train_labels, epochs):
    """Fresh training path -- needs labels, so it is separate from checkpoint load."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import emagerlib.models.new_emager_cnn as new_models

    model = getattr(new_models, model_name)(INPUT_SHAPE, num_classes)
    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_mav), torch.from_numpy(train_labels)),
        batch_size=64, shuffle=True, drop_last=True,
    )
    logger.info(f"  training {model_name} for {epochs} epoch(s) on {len(train_mav)} windows ...")
    model.fit(train_dl, max_epochs=epochs)
    model.eval().cpu()
    return model


def is_proto(model) -> bool:
    """True for the few-shot prototypical family (classify by distance to a
    `prototypes` buffer rather than through a learned classifier)."""
    return hasattr(model, "_prototypes_from")


def check_exportable(model, model_name):
    """
    Reject models that torch.onnx.export cannot handle, with a pointer at the fix.

    The torch.ao INT8 variants (…PTQ/…QAT) convert themselves during fit(), and eager-mode
    quantized modules have no ONNX lowering. They are also redundant here: this pipeline
    does its own INT8 via TFLiteConverter, calibrated on real MAV windows. Export the FP32
    variant and let stage 3 quantize it.
    """
    if getattr(model, "_quantized", False):
        fp32 = model_name.replace("QAT", "").replace("PTQ", "")
        raise SystemExit(
            f"\n{model_name} is a torch.ao-quantized model; its converted modules cannot be "
            f"exported to ONNX.\n"
            f"That chain targets PyTorch/FINN -- this tool quantizes to INT8 itself via "
            f"TFLiteConverter.\n"
            f"Export the FP32 variant instead and you get the same INT8 artifact:\n\n"
            f"    --model {fp32}\n"
        )


def bake_prototypes(model, train_mav, train_labels):
    """Compute generic 'factory' prototypes from the training reps and return them as
    a (C, D) float32 ndarray. A plain fit() with no test set leaves the buffer at zeros."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    proto_dl = DataLoader(
        TensorDataset(torch.from_numpy(train_mav), torch.from_numpy(train_labels)),
        batch_size=256,
    )
    model.prototypes = model._prototypes_from(proto_dl)
    model.eval().cpu()
    return model.prototypes.detach().cpu().numpy().astype(np.float32)


def embedding_only(model):
    """
    Wrap a prototypical model so forward() is embed(): (1, 64) -> (1, D).

    The prototype distance is deliberately left OFF the exported graph. Prototypes are
    user data computed on the device after flashing, so baking them into the network as
    folded constants would make on-device calibration impossible. The device runs this
    embedding in INT8 and does the distance itself against a writable array -- see
    c_runtime/emager_proto.h.

    This is the same boundary the PyTorch model already draws: the quantized proto
    variants bracket only embed() with Quant/DeQuant stubs and compute the distance in
    FP32.
    """
    import torch

    class _EmbeddingOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.embed(x)

    wrapper = _EmbeddingOnly(model)
    wrapper.eval()
    return wrapper


# ----------------------------------------------------------------------------
#  Reference logits (PyTorch, batch=1) -- ground truth for every verification
# ----------------------------------------------------------------------------

def torch_logits(model, x):
    """x: (N, 64) float32 ndarray -> (N, C) logits ndarray, evaluated one sample at a time."""
    import torch
    out = []
    with torch.no_grad():
        for row in x:
            out.append(model(torch.from_numpy(row[None, :])).cpu().numpy()[0])
    return np.stack(out)


def agreement(ref_logits, other_logits):
    """Top-1 agreement % and max abs logit difference between two (N, C) arrays."""
    top1 = float(np.mean(ref_logits.argmax(1) == other_logits.argmax(1)) * 100.0)
    maxd = float(np.max(np.abs(ref_logits - other_logits)))
    return top1, maxd


# ----------------------------------------------------------------------------
#  Stage 1: ONNX
# ----------------------------------------------------------------------------

def export_onnx(model, onnx_path, opset=17):
    import torch
    example = torch.zeros(1, N_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        model, example, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    logger.info(f"  [onnx]  wrote {onnx_path.name}  (opset {opset}, input [1,{N_FEATURES}])")


def verify_onnx(onnx_path, ref_logits, val_x, protos=None):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = np.stack([sess.run(None, {name: row[None, :].astype(np.float32)})[0][0] for row in val_x])
    if protos is not None:
        out = proto_logits(out, protos)   # graph emits embeddings; close the loop here
    return agreement(ref_logits, out)


# ----------------------------------------------------------------------------
#  Stage 2: ONNX -> TF SavedModel (+ float32 TFLite) via onnx2tf
#
#  onnx2tf's flatbuffer_direct path emits a proper TF SavedModel (needs the
#  optional tf_keras dep) plus a float32 .tflite. We take the SavedModel into
#  Stage 3 so TensorFlow's own TFLiteConverter does the INT8 calibration -- its
#  representative_dataset mechanism sets correct per-tensor input/output ranges
#  from real MAV windows, which onnx2tf's own -oiqt path does not (it defaults
#  activations to [-1, 1], clipping the real MAV/logit ranges to garbage).
# ----------------------------------------------------------------------------

def onnx_to_saved_model(onnx_path, work_dir):
    import onnx2tf
    sm = work_dir / "saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(sm),
        output_signaturedefs=True,
        flatbuffer_direct_output_saved_model=True,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    fp32 = next(iter(sm.glob("*float32.tflite")), None)
    if not (sm / "saved_model.pb").exists() or fp32 is None:
        raise RuntimeError(f"onnx2tf did not produce a SavedModel + float32 tflite in {sm}: "
                           f"{[p.name for p in sm.iterdir()]}")
    logger.info(f"  [tf]    onnx2tf -> SavedModel + {fp32.name}")
    return sm, fp32


# ----------------------------------------------------------------------------
#  Stage 3: full-INT8 TFLite (TFLiteConverter + representative dataset)
# ----------------------------------------------------------------------------

def convert_int8_tflite(saved_model_dir, out_path, calib_x):
    import tensorflow as tf

    def rep_gen():
        for row in calib_x:
            yield [row[None, :].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    out_path.write_bytes(data)
    logger.info(f"  [int8]  wrote {out_path.name}  ({len(data)/1024:.1f} KiB, full int8 I/O)")
    return out_path


def verify_tflite(tflite_path, ref_logits, val_x, protos=None):
    """
    Run a float or int8 .tflite one sample at a time; returns (top1%, maxdiff, quant_info).

    When `protos` is given the graph is the embedding only, so this reproduces the whole
    on-device path exactly as emager_proto.c does it:

        float MAV -> int8 quantize -> TFLite -> dequantize -> prototype distance -> argmax

    and still compares against the PyTorch model's complete forward. Checking the
    embedding tensor alone would say nothing about whether the device predicts the same
    class, which is the only thing that matters.
    """
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp   = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    is_int8 = inp["dtype"] == np.int8

    outputs = []
    for row in val_x:
        x = row[None, :].astype(np.float32)
        if is_int8:
            x = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - out_zp) * out_scale
        outputs.append(y)

    logits = np.stack(outputs)
    if protos is not None:
        logits = proto_logits(logits, protos)
    top1, maxd = agreement(ref_logits, logits)
    quant = {"in_scale": in_scale, "in_zp": in_zp, "out_scale": out_scale,
             "out_zp": out_zp, "int8_io": is_int8}
    return top1, maxd, quant


# ----------------------------------------------------------------------------
#  Stage 4: C array (TFLite Micro style header)
# ----------------------------------------------------------------------------

def proto_logits(embeddings, protos):
    """
    Prototype distance, in numpy, matching BOTH the PyTorch forward and emager_proto.c:
        logit_c = -||e - p_c||^2 / D

    embeddings: (N, D)   protos: (C, D)  ->  (N, C)
    """
    d = ((embeddings[:, None, :] - protos[None, :, :]) ** 2).sum(-1)
    return -d / embeddings.shape[1]


def write_model_params_header(path, n_features, embed_dim, n_classes, quant, model_name):
    """Dimensions + INT8 scale/zero-point, read off the actual .tflite."""
    path.write_text("\n".join([
        "/* Auto-generated by examples/deployment/export_model.py -- do not edit.",
        f" * Model: {model_name}",
        " *",
        " * The scale/zero-point values are read from the exported .tflite, not chosen.",
        " * Re-export if you retrain: they change with the calibration data.",
        " */",
        "#ifndef EMAGER_MODEL_PARAMS_H",
        "#define EMAGER_MODEL_PARAMS_H",
        "",
        "/* Input: MAV over a 200-sample window of the 4x16 electrode grid. */",
        f"#define EMAGER_N_FEATURES        {n_features}",
        "/* Output of the exported network: the embedding f_theta(x). */",
        f"#define EMAGER_EMBED_DIM         {embed_dim}",
        f"#define EMAGER_N_CLASSES         {n_classes}",
        "",
        "/* TFLite affine quantization: real = scale * (q - zero_point) */",
        f"#define EMAGER_INPUT_SCALE       {quant['in_scale']:.10g}f",
        f"#define EMAGER_INPUT_ZERO_POINT  ({int(quant['in_zp'])})",
        f"#define EMAGER_EMBED_SCALE       {quant['out_scale']:.10g}f",
        f"#define EMAGER_EMBED_ZERO_POINT  ({int(quant['out_zp'])})",
        "",
        "#endif /* EMAGER_MODEL_PARAMS_H */",
        "",
    ]), encoding="ascii")
    logger.info(f"  [c]     wrote {path.name}  (dims + int8 scale/zero-point)")


def write_prototypes_header(path, protos, model_name, dataset, source_note):
    """
    Factory prototypes as a compile-time default.

    Deliberately `static const`: these are the shipped fallback, computed at the factory
    from training data. The device overwrites them at runtime with the user's own
    calibration (into a mutable emager_prototypes_t) -- that copy is what gets persisted.
    """
    n_classes, embed_dim = protos.shape
    lines = [
        "/* Auto-generated by examples/deployment/export_model.py -- do not edit.",
        f" * Factory prototypes for {model_name}",
        f" * Source: {source_note}",
        f" * Dataset: {dataset}",
        " *",
        " * These are the 'shipped' defaults: generic prototypes computed at the factory,",
        " * good enough to work uncalibrated. On-device calibration REPLACES them with the",
        " * user's own (see emager_calib_* in emager_proto.h) -- copy this into a mutable",
        " * emager_prototypes_t first, do not try to write through this const.",
        " */",
        "#ifndef EMAGER_PROTOTYPES_H",
        "#define EMAGER_PROTOTYPES_H",
        "",
        '#include "emager_proto.h"',
        "",
        "static const emager_prototypes_t EMAGER_FACTORY_PROTOTYPES = {",
        "    .v = {",
    ]
    for c in range(n_classes):
        lines.append(f"        /* class {c} */ {{")
        row = protos[c]
        for i in range(0, embed_dim, 6):
            chunk = ", ".join(f"{v: .8e}f" for v in row[i:i + 6])
            lines.append(f"            {chunk},")
        lines.append("        },")
    lines += [
        "    },",
        "    .valid = { " + ", ".join("1" for _ in range(n_classes)) + " },",
        "};",
        "",
        "#endif /* EMAGER_PROTOTYPES_H */",
        "",
    ]
    # ascii, deliberately: these headers go into embedded toolchains, and encoding="ascii"
    # fails here at generation time rather than on the target's compiler.
    path.write_text("\n".join(lines), encoding="ascii")
    logger.info(f"  [c]     wrote {path.name}  (factory prototypes, {n_classes}x{embed_dim})")


def copy_c_runtime(out_dir):
    """Copy the hand-written C runtime next to the generated headers so the output dir
    is a self-contained drop-in for a firmware project."""
    import shutil
    src = Path(__file__).parent / "c_runtime"
    copied = []
    # emager_selftest.* is copied too, but it will not compile until make_test_vectors.py
    # has written emager_test_vectors.h next to it -- that header needs this export's
    # weights, so it cannot be produced here.
    for pattern in ("emager_proto.*", "emager_selftest.*"):
        for f in sorted(src.glob(pattern)):
            shutil.copy2(f, out_dir / f.name)
            copied.append(f.name)
    logger.info(f"  [c]     copied runtime: {', '.join(copied)}")
    return copied


def tflite_to_c_array(tflite_path, header_path, var_name="g_model"):
    data = tflite_path.read_bytes()
    lines = [
        "// Auto-generated by examples/deployment/export_model.py -- do not edit.",
        f"// Source: {tflite_path.name}  ({len(data)} bytes)",
        "#include <cstdint>", "",
        f"alignas(8) const unsigned char {var_name}[] = {{",
    ]
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i:i + 12])
        lines.append(f"  {chunk},")
    lines += ["};", f"const unsigned int {var_name}_len = {len(data)};", ""]
    header_path.write_text("\n".join(lines), encoding="ascii")
    logger.info(f"  [carray] wrote {header_path.name}  ({len(data)} bytes -> {var_name}[])")
    return header_path


# ----------------------------------------------------------------------------
#  Orchestration
# ----------------------------------------------------------------------------

def run(args):
    out_dir = Path(args.out) if args.out else (Path(__file__).parent / "exported" / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work"
    work_dir.mkdir(exist_ok=True)

    formats = set(args.formats)
    report = [f"# Export report -- {args.model}", ""]

    # -- data + model --------------------------------------------------------
    logger.info("Loading data ...")
    from train_cnn_benchmark import BASE_PATH, load_and_filter, read_dataset_info
    path = BASE_PATH / args.dataset
    num_classes, num_reps = read_dataset_info(path)
    odh = load_and_filter(path, num_classes, num_reps, SAMPLING)
    held_out   = num_reps - 1
    train_reps = [r for r in range(num_reps) if r != held_out]
    tr_win, tr_meta = odh.isolate_data("reps", train_reps).parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    va_win, va_meta = odh.isolate_data("reps", [held_out]).parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    train_mav    = np.mean(np.abs(tr_win), axis=2).astype(np.float32)
    train_labels = tr_meta["classes"].astype(np.int64)
    val_mav      = np.mean(np.abs(va_win), axis=2).astype(np.float32)
    val_labels   = va_meta["classes"].astype(np.int64)
    logger.info(f"  dataset={args.dataset} classes={num_classes} reps={num_reps} "
                f"train {train_mav.shape} val {val_mav.shape}")

    logger.info("Preparing model ...")
    if args.checkpoint:
        model = load_checkpoint_model(args.model, num_classes, args.checkpoint)
    else:
        model = train_model(args.model, num_classes, train_mav, train_labels, args.epochs)
        # Keep the weights that produced this export. Without this the FP32 source of a
        # shipped artifact is unreproducible -- training is not seeded, so re-running the
        # tool gives a different model, and the EXPORT_REPORT's accuracy would then belong
        # to weights that no longer exist anywhere. Re-export byte-identically with
        # --checkpoint <this file>.
        import torch
        ckpt = out_dir / "model_fp32.pt"
        torch.save(model.state_dict(), ckpt)
        logger.info(f"  saved FP32 weights -> {ckpt.name}  (re-export with --checkpoint)")

    check_exportable(model, args.model)

    # Few-shot prototypical models classify by distance to a `prototypes` buffer that a
    # plain fit() (no test set) leaves at zeros. Compute generic "factory" prototypes
    # from the training reps either way -- in `embedding` mode they are emitted as a C
    # header, in `baked` mode they are folded into the graph.
    protos = None
    proto_mode = args.proto_export
    if is_proto(model):
        protos = bake_prototypes(model, train_mav, train_labels)
        logger.info(f"  computed generic prototypes from training reps  (shape {protos.shape})")
        logger.info(f"  prototype export mode: {proto_mode}"
                    + ("  (embedding on the graph, prototypes writable in C)"
                       if proto_mode == "embedding"
                       else "  (prototypes folded into the graph -- NOT recalibratable)"))
    elif proto_mode == "embedding":
        # Not a proto model: nothing to split off, so fall back silently to the whole graph.
        proto_mode = "baked"

    # Verification / calibration subsets (cap for speed).
    rng = np.random.default_rng(0)
    val_idx   = rng.permutation(len(val_mav))[: args.verify_samples]
    verify_x, verify_y = val_mav[val_idx], val_labels[val_idx]
    calib_idx = rng.permutation(len(train_mav))[: args.calib_samples]
    calib_x   = train_mav[calib_idx]

    # Reference is always the model's FULL forward (embedding + prototype distance),
    # so `embedding` mode is held to the same standard as `baked`: the device must
    # predict the same class, not merely produce a similar embedding.
    ref_logits = torch_logits(model, verify_x)
    torch_acc  = float(np.mean(ref_logits.argmax(1) == verify_y) * 100.0)
    logger.info(f"  PyTorch reference top-1 on held-out rep: {torch_acc:.1f}%  "
                f"({len(verify_x)} windows)")
    report += [f"- Model: `{args.model}`  |  Dataset: `{args.dataset}`"]
    if is_proto(model):
        report += [f"- Prototype export mode: **{proto_mode}**"
                   + ("  — the graph is the embedding `f_θ` only; prototypes live in "
                      "`emager_prototypes.h` and are rewritable on device."
                      if proto_mode == "embedding"
                      else "  — prototypes folded into the graph; **not** recalibratable on device.")]
    report += [f"- PyTorch reference accuracy (held-out rep): **{torch_acc:.1f}%** "
               f"on {len(verify_x)} windows", ""]
    report += ["| Artifact | Size | Top-1 agreement vs PyTorch | Max logit diff |",
               "|---|---|---|---|"]

    onnx_path = out_dir / "model.onnx"

    # In `embedding` mode everything downstream converts the embedding-only graph, and
    # every verification has to re-attach the prototype distance to get back to logits.
    embedding_mode = is_proto(model) and proto_mode == "embedding"
    export_target  = embedding_only(model) if embedding_mode else model
    verify_protos  = protos if embedding_mode else None

    # -- Stage 1: ONNX -------------------------------------------------------
    need_onnx = formats & {"onnx", "tflite-fp32", "tflite-int8", "carray"}
    if need_onnx:
        logger.info("Stage 1: ONNX export ...")
        export_onnx(export_target, onnx_path)
        if "onnx" in formats:
            top1, maxd = verify_onnx(onnx_path, ref_logits, verify_x, protos=verify_protos)
            logger.info(f"  [onnx]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            report.append(f"| model.onnx | {onnx_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")

    # -- Stage 2: SavedModel + float32 TFLite --------------------------------
    int8_path = out_dir / "model_int8.tflite"
    if formats & {"tflite-fp32", "tflite-int8", "carray"}:
        logger.info("Stage 2: ONNX -> TF SavedModel + float32 TFLite (onnx2tf) ...")
        # onnx2tf runs its simplifier over the input ONNX and rewrites it IN PLACE
        # (~9 KiB smaller). Left alone, that would swap the shipped model.onnx for one
        # that was never verified -- verify_onnx() ran against the pre-simplification
        # graph. Hand onnx2tf a scratch copy so the artifact on disk stays exactly the
        # bytes the report vouches for.
        import shutil
        onnx_for_tf = work_dir / "model_for_tf.onnx"
        shutil.copy2(onnx_path, onnx_for_tf)
        saved_model_dir, fp32_src = onnx_to_saved_model(onnx_for_tf, work_dir)

        if "tflite-fp32" in formats:
            fp32_path = out_dir / "model_float32.tflite"
            fp32_path.write_bytes(fp32_src.read_bytes())
            top1, maxd, _ = verify_tflite(fp32_path, ref_logits, verify_x, protos=verify_protos)
            logger.info(f"  [fp32]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            report.append(f"| model_float32.tflite | {fp32_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")

        # -- Stage 3: full-INT8 TFLite ---------------------------------------
        if formats & {"tflite-int8", "carray"}:
            logger.info("Stage 3: full-INT8 TFLite (representative-dataset calibration) ...")
            convert_int8_tflite(saved_model_dir, int8_path, calib_x)
            top1, maxd, quant = verify_tflite(int8_path, ref_logits, verify_x, protos=verify_protos)
            logger.info(f"  [int8]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            logger.info(f"  [int8]  input  scale={quant['in_scale']:.6g} zero_point={quant['in_zp']}")
            logger.info(f"  [int8]  output scale={quant['out_scale']:.6g} zero_point={quant['out_zp']}")
            report.append(f"| model_int8.tflite | {int8_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")
            out_label = "embedding" if embedding_mode else "logit"
            report += ["",
                       "## INT8 on-device I/O quantization",
                       "TFLite affine quant: `real = scale * (q_int8 - zero_point)`.",
                       f"- input : `q = round(mav / {quant['in_scale']:.8g}) + ({quant['in_zp']})`",
                       f"- output: `{out_label} = (q - ({quant['out_zp']})) * {quant['out_scale']:.8g}`",
                       "",
                       "`emager_proto.c` already does both of these for you "
                       "(`emager_quantize_input` / `emager_dequantize_embedding`) using the "
                       "constants in `emager_model_params.h`.", ""]

    # -- Stage 4: C array ----------------------------------------------------
    if "carray" in formats:
        logger.info("Stage 4: C array header ...")
        tflite_to_c_array(int8_path, out_dir / "emager_model_data.h")

    # -- Stage 5: C runtime + generated headers (prototypical models) ---------
    if embedding_mode and formats & {"tflite-int8", "carray"}:
        logger.info("Stage 5: C runtime + generated headers ...")
        write_model_params_header(out_dir / "emager_model_params.h", N_FEATURES,
                                  int(protos.shape[1]), int(protos.shape[0]), quant, args.model)
        write_prototypes_header(out_dir / "emager_prototypes.h", protos, args.model,
                                args.dataset, "generic prototypes, mean embedding per class "
                                              "over the training reps")
        copy_c_runtime(out_dir)
        report += ["## On-device prototypes",
                   "",
                   f"The graph is the embedding `f_θ` only — `[1,{N_FEATURES}] → "
                   f"[1,{protos.shape[1]}]`. The prototype distance is **not** on the graph, "
                   "so the prototypes stay writable after flashing and the device can "
                   "recalibrate them from the user's own gestures (no backprop — just a "
                   "forward pass and a per-class mean).",
                   "",
                   "| File | Role |",
                   "|---|---|",
                   "| `emager_model_data.h` | the INT8 embedding network, as a C byte array |",
                   "| `emager_model_params.h` | dims + INT8 scale/zero-point (generated from this .tflite) |",
                   "| `emager_prototypes.h` | factory prototypes — the shipped default |",
                   "| `emager_proto.h` / `.c` | classify + on-device calibration |",
                   "",
                   f"Prototype storage: `{protos.shape[0]}×{protos.shape[1]}` float32 + "
                   f"{protos.shape[0]} validity flags = "
                   f"**{protos.shape[0] * protos.shape[1] * 4 + protos.shape[0]} bytes** of data to "
                   "persist after calibration. The struct is a little larger once padded — "
                   "`EMAGER_PROTOTYPES_NBYTES` in `emager_proto.h` is the authoritative size.",
                   "",
                   "See [`../../README.md`](../../README.md) for the firmware integration guide.", ""]

    # utf-8, not the platform default: the report contains non-ASCII (f_θ, ×) and
    # Windows would otherwise default to cp1252 and raise on it.
    (out_dir / "EXPORT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if not args.keep_work and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    logger.info(f"Done. Artifacts + EXPORT_REPORT.md in {out_dir}")
    print("\n".join(report))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export an EMaGer CNN to MCU-ready ONNX/TFLite/C-array.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model class in new_emager_cnn.py")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset folder under Datasets/")
    p.add_argument("--checkpoint", default=None, help="path to a state_dict .pt (skip training)")
    p.add_argument("--epochs", type=int, default=10, help="training epochs when no checkpoint")
    p.add_argument("--formats", nargs="+", default=["onnx", "tflite-fp32", "tflite-int8", "carray"],
                   choices=["onnx", "tflite-fp32", "tflite-int8", "carray"])
    p.add_argument("--proto-export", default="embedding", choices=["embedding", "baked"],
                   help="prototypical models only. 'embedding' (default) exports f_theta alone "
                        "and emits the prototypes as a C header, so the device can recalibrate "
                        "them from the user's gestures. 'baked' folds the prototypes into the "
                        "graph: self-contained, but frozen at flash time. Ignored for "
                        "non-prototypical models.")
    p.add_argument("--calib-samples", type=int, default=300, help="representative windows for int8")
    p.add_argument("--verify-samples", type=int, default=256, help="held-out windows for verification")
    p.add_argument("--out", default=None, help="output dir (default: examples/deployment/exported/<model>)")
    p.add_argument("--keep-work", action="store_true", help="keep the _work/ intermediates (SavedModel etc.)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        # Windows consoles default to cp1252, which cannot encode the report's f_θ / ×.
        # Same guard as eval_fewshot_loso.py.
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")
    run(args)


if __name__ == "__main__":
    main()
