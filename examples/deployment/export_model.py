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

Usage
-----
    python examples/deployment/export_model.py                       # RingStrided, Test_EM_C7_R5
    python examples/deployment/export_model.py --model EmagerCNNRingStrided --dataset Test_EM_C7_R5
    python examples/deployment/export_model.py --checkpoint path/state_dict.pt --model EmagerCNNBase
    python examples/deployment/export_model.py --formats onnx tflite-int8 carray
    python examples/deployment/export_model.py --epochs 10 --calib-samples 300

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


def verify_onnx(onnx_path, ref_logits, val_x):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = np.stack([sess.run(None, {name: row[None, :].astype(np.float32)})[0][0] for row in val_x])
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


def verify_tflite(tflite_path, ref_logits, val_x):
    """Run a float or int8 .tflite one sample at a time; returns (top1%, maxdiff, quant_info)."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp   = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    is_int8 = inp["dtype"] == np.int8

    logits = []
    for row in val_x:
        x = row[None, :].astype(np.float32)
        if is_int8:
            x = np.round(x / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            y = (y - out_zp) * out_scale
        logits.append(y)
    top1, maxd = agreement(ref_logits, np.stack(logits))
    quant = {"in_scale": in_scale, "in_zp": in_zp, "out_scale": out_scale,
             "out_zp": out_zp, "int8_io": is_int8}
    return top1, maxd, quant


# ----------------------------------------------------------------------------
#  Stage 4: C array (TFLite Micro style header)
# ----------------------------------------------------------------------------

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
    header_path.write_text("\n".join(lines))
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

    # Few-shot prototypical models classify by distance to a `prototypes` buffer
    # that a plain fit() (no test set) leaves at zeros. Bake in generic "factory"
    # prototypes from the training reps so the exported forward is self-contained.
    if hasattr(model, "_prototypes_from"):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        proto_dl = DataLoader(
            TensorDataset(torch.from_numpy(train_mav), torch.from_numpy(train_labels)),
            batch_size=256,
        )
        model.prototypes = model._prototypes_from(proto_dl)
        model.eval().cpu()
        logger.info(f"  baked generic prototypes from training reps  "
                    f"(shape {tuple(model.prototypes.shape)})")

    # Verification / calibration subsets (cap for speed).
    rng = np.random.default_rng(0)
    val_idx   = rng.permutation(len(val_mav))[: args.verify_samples]
    verify_x, verify_y = val_mav[val_idx], val_labels[val_idx]
    calib_idx = rng.permutation(len(train_mav))[: args.calib_samples]
    calib_x   = train_mav[calib_idx]

    ref_logits = torch_logits(model, verify_x)
    torch_acc  = float(np.mean(ref_logits.argmax(1) == verify_y) * 100.0)
    logger.info(f"  PyTorch reference top-1 on held-out rep: {torch_acc:.1f}%  "
                f"({len(verify_x)} windows)")
    report += [f"- PyTorch reference accuracy (held-out rep): **{torch_acc:.1f}%** "
               f"on {len(verify_x)} windows", ""]
    report += ["| Artifact | Size | Top-1 agreement vs PyTorch | Max logit diff |",
               "|---|---|---|---|"]

    onnx_path = out_dir / "model.onnx"

    # -- Stage 1: ONNX -------------------------------------------------------
    need_onnx = formats & {"onnx", "tflite-fp32", "tflite-int8", "carray"}
    if need_onnx:
        logger.info("Stage 1: ONNX export ...")
        export_onnx(model, onnx_path)
        if "onnx" in formats:
            top1, maxd = verify_onnx(onnx_path, ref_logits, verify_x)
            logger.info(f"  [onnx]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            report.append(f"| model.onnx | {onnx_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")

    # -- Stage 2: SavedModel + float32 TFLite --------------------------------
    int8_path = out_dir / "model_int8.tflite"
    if formats & {"tflite-fp32", "tflite-int8", "carray"}:
        logger.info("Stage 2: ONNX -> TF SavedModel + float32 TFLite (onnx2tf) ...")
        saved_model_dir, fp32_src = onnx_to_saved_model(onnx_path, work_dir)

        if "tflite-fp32" in formats:
            fp32_path = out_dir / "model_float32.tflite"
            fp32_path.write_bytes(fp32_src.read_bytes())
            top1, maxd, _ = verify_tflite(fp32_path, ref_logits, verify_x)
            logger.info(f"  [fp32]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            report.append(f"| model_float32.tflite | {fp32_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")

        # -- Stage 3: full-INT8 TFLite ---------------------------------------
        if formats & {"tflite-int8", "carray"}:
            logger.info("Stage 3: full-INT8 TFLite (representative-dataset calibration) ...")
            convert_int8_tflite(saved_model_dir, int8_path, calib_x)
            top1, maxd, quant = verify_tflite(int8_path, ref_logits, verify_x)
            logger.info(f"  [int8]  agreement {top1:.1f}%  maxdiff {maxd:.2e}")
            logger.info(f"  [int8]  input  scale={quant['in_scale']:.6g} zero_point={quant['in_zp']}")
            logger.info(f"  [int8]  output scale={quant['out_scale']:.6g} zero_point={quant['out_zp']}")
            report.append(f"| model_int8.tflite | {int8_path.stat().st_size/1024:.1f} KiB | "
                          f"{top1:.1f}% | {maxd:.2e} |")
            report += ["",
                       "## INT8 on-device I/O quantization",
                       "TFLite affine quant: `real = scale * (q_int8 - zero_point)`.",
                       f"- input : `q = round(mav / {quant['in_scale']:.8g}) + ({quant['in_zp']})`",
                       f"- output: `logit = (q - ({quant['out_zp']})) * {quant['out_scale']:.8g}`", ""]

    # -- Stage 4: C array ----------------------------------------------------
    if "carray" in formats:
        logger.info("Stage 4: C array header ...")
        tflite_to_c_array(int8_path, out_dir / "model_int8.h")

    (out_dir / "EXPORT_REPORT.md").write_text("\n".join(report))
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
    p.add_argument("--calib-samples", type=int, default=300, help="representative windows for int8")
    p.add_argument("--verify-samples", type=int, default=256, help="held-out windows for verification")
    p.add_argument("--out", default=None, help="output dir (default: examples/deployment/exported/<model>)")
    p.add_argument("--keep-work", action="store_true", help="keep the _work/ intermediates (SavedModel etc.)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")
    run(args)


if __name__ == "__main__":
    main()
