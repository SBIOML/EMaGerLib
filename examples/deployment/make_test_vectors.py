"""
Generate on-device test vectors for a previously exported EMaGer model.

Why this exists
---------------
`export_model.py` verifies the artifacts against PyTorch *on the host*. It cannot
verify your firmware: whether the MAV features are computed the way training did,
whether the input quantization is right, whether the NPU/CMSIS kernels reproduce the
reference. That check has to happen on the board, and it needs reference data that was
produced by the exact model that got flashed.

This tool emits `emager_test_vectors.h`: a handful of real MAV windows together with
the embedding and class the host says they should produce. `emager_selftest.c` replays
them through your firmware's forward pass and reports where -- if anywhere -- the
device diverges.

    float MAV[64]  ->  emager_quantize_input  ->  <your NN>  ->  int8 embedding
                                                                     |
                                            compared against EMAGER_TEST_EMBED[i]

Comparing the *embedding* rather than only the class is the point. A class mismatch
tells you something is wrong; an embedding mismatch tells you it is the network path
(features, quantization, kernels) rather than the prototypes. That is the split the
deployment README's "Verifying the port" section asks you to make, and you cannot make
it without a reference embedding.

Usage
-----
    # after an export
    python examples/deployment/make_test_vectors.py

    # explicit
    python examples/deployment/make_test_vectors.py \
        --export-dir examples/deployment/exported/EmagerCNNProtoRingStrided --num 32

The export directory must still contain `model_int8.tflite`, `model_fp32.pt`,
`emager_model_params.h` and (for prototypical models) `emager_prototypes.h` -- the
vectors are only meaningful for that exact set of weights.

Requires the same "deploy" optional-dependency group as export_model.py.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = logging.getLogger(__name__)

# Reuse the export tool's pipeline wholesale: the vectors must be produced by the same
# windowing and feature code the exported model was calibrated on, so importing is the
# only safe option -- a second copy of these constants would drift.
sys.path.insert(0, str(Path(__file__).parent))
from export_model import (  # noqa: E402
    N_FEATURES, WINDOW_SIZE, WINDOW_INCREMENT, SAMPLING,
    DEFAULT_MODEL, DEFAULT_DATASET,
    load_checkpoint_model, is_proto, torch_logits, proto_logits,
)

DEFAULT_NUM_VECTORS = 32
# Slack allowed between the host reference embedding and the device's, in INT8 LSBs.
# Not zero on purpose: the host runs TFLite's reference kernels while the device runs
# CMSIS-NN or the Neural-ART NPU. Those agree on the arithmetic but not always on the
# last bit of rounding, and demanding bit-exactness would fail a correct port.
DEFAULT_TOL_LSB = 2


# ----------------------------------------------------------------------------
#  Reading back what was exported
# ----------------------------------------------------------------------------

def read_params_header(path):
    """Pull the dims and quantization constants back out of emager_model_params.h.

    Read from the generated header rather than recomputed: these are the numbers the
    firmware is compiled against, so anything else would be checking a different model
    than the one on the board.
    """
    text = path.read_text(encoding="ascii")
    out = {}
    for key, pattern in {
        "n_features": r"#define\s+EMAGER_N_FEATURES\s+(\d+)",
        "embed_dim":  r"#define\s+EMAGER_EMBED_DIM\s+(\d+)",
        "n_classes":  r"#define\s+EMAGER_N_CLASSES\s+(\d+)",
    }.items():
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"{path.name}: could not find {key}")
        out[key] = int(m.group(1))
    for key, pattern in {
        "embed_scale": r"#define\s+EMAGER_EMBED_SCALE\s+([-\d.eE+]+)f",
        "input_scale": r"#define\s+EMAGER_INPUT_SCALE\s+([-\d.eE+]+)f",
    }.items():
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"{path.name}: could not find {key}")
        out[key] = float(m.group(1))
    for key, pattern in {
        "embed_zp": r"#define\s+EMAGER_EMBED_ZERO_POINT\s+\((-?\d+)\)",
        "input_zp": r"#define\s+EMAGER_INPUT_ZERO_POINT\s+\((-?\d+)\)",
    }.items():
        m = re.search(pattern, text)
        if not m:
            raise ValueError(f"{path.name}: could not find {key}")
        out[key] = int(m.group(1))
    return out


def read_prototypes_header(path, n_classes, embed_dim):
    """Parse the factory prototypes back out of emager_prototypes.h.

    Recomputing them from the checkpoint would give the same numbers, but only as long
    as nobody edits the header or re-bakes with different data. Parsing what is actually
    compiled in removes that assumption.
    """
    text = path.read_text(encoding="ascii")
    body = text.split(".v = {", 1)[1].split(".valid", 1)[0]
    vals = [float(v) for v in re.findall(r"(-?\d+\.\d+e[+-]\d+)f", body)]
    expected = n_classes * embed_dim
    if len(vals) != expected:
        raise ValueError(f"{path.name}: parsed {len(vals)} floats, expected {expected}")
    return np.asarray(vals, dtype=np.float32).reshape(n_classes, embed_dim)


# ----------------------------------------------------------------------------
#  Sampling
# ----------------------------------------------------------------------------

def stratified_pick(labels, n_classes, num):
    """Choose `num` window indices spread evenly over the classes.

    Uniform random sampling can miss a gesture entirely on an unbalanced held-out rep,
    and a self-test that never exercises a class is exactly the one that passes while
    that class is broken. Deterministic (no RNG) so re-running gives the same header.
    """
    if num < n_classes:
        logger.warning(f"  --num {num} is below the {n_classes} classes: {n_classes - num} "
                       f"gesture(s) will not be exercised at all. A self-test that skips a "
                       f"class is the one that passes while that class is broken.")
    base, extra = divmod(num, n_classes)
    picked = []
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            logger.warning(f"  class {c} has no held-out windows -- not covered by the self-test")
            continue
        # Hand the remainder to the first `extra` classes so `--num` means what it says
        # instead of silently rounding down to a multiple of the class count.
        want = max(1, base + (1 if c < extra else 0))
        # Spread across the rep rather than taking the first N: consecutive windows
        # overlap by WINDOW_SIZE-WINDOW_INCREMENT samples and are nearly identical.
        step = max(1, idx.size // want)
        picked.extend(idx[::step][:want].tolist())
    return np.asarray(sorted(picked)[:num], dtype=np.int64)


# ----------------------------------------------------------------------------
#  Running the exported network exactly as the device will
# ----------------------------------------------------------------------------

def run_int8_tflite(tflite_path, mav):
    """Return (raw int8 embeddings, dequantized float embeddings) for each MAV row.

    Mirrors emager_quantize_input / emager_dequantize_embedding rather than letting the
    interpreter take floats, so the reference includes the quantization step the
    firmware performs.
    """
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    if inp["dtype"] != np.int8 or out["dtype"] != np.int8:
        raise ValueError(f"{tflite_path.name} is not a full-INT8 model -- export with tflite-int8")

    raw, deq = [], []
    for row in mav:
        q = np.round(row[None, :] / in_scale + in_zp).clip(-128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0].astype(np.int8)
        raw.append(y)
        deq.append((y.astype(np.float32) - out_zp) * out_scale)
    return np.stack(raw), np.stack(deq)


# ----------------------------------------------------------------------------
#  Header emission
# ----------------------------------------------------------------------------

def write_test_vectors_header(path, mav, embed_q, classes, model_name, dataset,
                              tol_lsb, host_top1):
    n, n_features = mav.shape
    embed_dim = embed_q.shape[1]
    lines = [
        "/* Auto-generated by examples/deployment/make_test_vectors.py -- do not edit.",
        f" * Model: {model_name}   Dataset: {dataset}",
        f" * {n} held-out windows, stratified over the classes.",
        " *",
        " * EMAGER_TEST_MAV[i]   -- input, exactly as emager_predict() expects it",
        " * EMAGER_TEST_EMBED[i] -- the INT8 embedding the host's model_int8.tflite produced",
        " * EMAGER_TEST_CLASS[i] -- the class the FACTORY prototypes give for that embedding",
        " *",
        " * The class column is only valid BEFORE on-device calibration: calibrating",
        " * replaces the prototypes, and the same embedding then legitimately maps to a",
        " * different class. The embedding column stays valid for the life of the flashed",
        " * network -- that is the one to trust when diagnosing a port.",
        " *",
        f" * Host agreement of this INT8 model against the FP32 PyTorch model: {host_top1:.1f}%",
        " * (an imperfect number here is quantization loss, not a firmware bug -- it is the",
        " *  ceiling the device is being measured against, not a target to beat).",
        " */",
        "#ifndef EMAGER_TEST_VECTORS_H",
        "#define EMAGER_TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        '#include "emager_model_params.h"',
        "",
        f"#define EMAGER_TEST_N_VECTORS    {n}",
        "/* Tolerance on the embedding, in INT8 LSBs. The host runs TFLite's reference",
        " * kernels; the device runs CMSIS-NN or the Neural-ART NPU. They agree on the",
        " * arithmetic but not always on the last bit of rounding. */",
        f"#define EMAGER_TEST_EMBED_TOL    {tol_lsb}",
        "",
    ]

    if n_features != N_FEATURES or embed_dim <= 0:
        raise ValueError("dimension mismatch between the vectors and the exported model")

    lines.append("static const float EMAGER_TEST_MAV[EMAGER_TEST_N_VECTORS][EMAGER_N_FEATURES] = {")
    for i in range(n):
        lines.append(f"    /* [{i}] */ {{")
        for j in range(0, n_features, 6):
            chunk = ", ".join(f"{v: .8e}f" for v in mav[i, j:j + 6])
            lines.append(f"        {chunk},")
        lines.append("    },")
    lines += ["};", ""]

    lines.append("static const int8_t EMAGER_TEST_EMBED[EMAGER_TEST_N_VECTORS][EMAGER_EMBED_DIM] = {")
    for i in range(n):
        row = ", ".join(f"{int(v):4d}" for v in embed_q[i])
        lines.append(f"    /* [{i}] */ {{ {row} }},")
    lines += ["};", ""]

    lines.append("static const uint8_t EMAGER_TEST_CLASS[EMAGER_TEST_N_VECTORS] = {")
    for i in range(0, n, 16):
        lines.append("    " + ", ".join(str(int(c)) for c in classes[i:i + 16]) + ",")
    lines += ["};", "", "#endif /* EMAGER_TEST_VECTORS_H */", ""]

    # ascii on purpose, same reason as the other generated headers: an encoding mistake
    # must fail here and not on the target's compiler.
    path.write_text("\n".join(lines), encoding="ascii")
    approx_bytes = n * (n_features * 4 + embed_dim + 1)
    logger.info(f"  wrote {path.name}  ({n} vectors, ~{approx_bytes / 1024:.1f} KiB of const flash)")


# ----------------------------------------------------------------------------
#  Orchestration
# ----------------------------------------------------------------------------

def run(args):
    export_dir = Path(args.export_dir) if args.export_dir else \
        (Path(__file__).parent / "exported" / args.model)
    if not export_dir.is_dir():
        raise SystemExit(f"no export directory at {export_dir} -- run export_model.py first")

    tflite = export_dir / "model_int8.tflite"
    params_h = export_dir / "emager_model_params.h"
    ckpt = Path(args.checkpoint) if args.checkpoint else (export_dir / "model_fp32.pt")
    for f in (tflite, params_h, ckpt):
        if not f.exists():
            raise SystemExit(f"missing {f} -- re-run export_model.py (it writes all of these)")

    params = read_params_header(params_h)
    logger.info(f"Export: {export_dir}")
    logger.info(f"  {params['n_features']} features -> {params['embed_dim']}-d embedding, "
                f"{params['n_classes']} classes")

    # -- data ---------------------------------------------------------------
    logger.info("Loading data ...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
    from train_cnn_benchmark import BASE_PATH, load_and_filter, read_dataset_info
    path = BASE_PATH / args.dataset
    num_classes, num_reps = read_dataset_info(path)
    if num_classes != params["n_classes"]:
        raise SystemExit(f"dataset has {num_classes} classes but the export was built for "
                         f"{params['n_classes']} -- wrong --dataset?")
    odh = load_and_filter(path, num_classes, num_reps, SAMPLING)
    # Same held-out rep as export_model.py, so the vectors are windows the exported
    # model was never trained on.
    held_out = num_reps - 1
    va_win, va_meta = odh.isolate_data("reps", [held_out]).parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
    val_mav = np.mean(np.abs(va_win), axis=2).astype(np.float32)
    val_labels = va_meta["classes"].astype(np.int64)

    pick = stratified_pick(val_labels, num_classes, args.num)
    mav = val_mav[pick]
    logger.info(f"  picked {len(pick)} windows from rep {held_out} "
                f"(classes covered: {sorted(set(val_labels[pick].tolist()))})")

    # -- reference: the FP32 PyTorch model, full forward ---------------------
    logger.info("Running the PyTorch reference ...")
    model = load_checkpoint_model(args.model, num_classes, ckpt)
    protos = None
    if is_proto(model):
        protos_h = export_dir / "emager_prototypes.h"
        if not protos_h.exists():
            raise SystemExit(f"missing {protos_h} -- this is a prototypical model")
        protos = read_prototypes_header(protos_h, num_classes, params["embed_dim"])
        logger.info(f"  factory prototypes: {protos.shape} (parsed from {protos_h.name})")
        # Load them into the model so its forward matches what the device computes.
        import torch
        with torch.no_grad():
            model.prototypes.copy_(torch.from_numpy(protos))
    ref_logits = torch_logits(model, mav)

    # -- what the device should produce -------------------------------------
    logger.info("Running the exported INT8 model ...")
    # The .tflite already IS the embedding for prototypical exports (--proto-export
    # embedding), so the prototype distance is re-attached here in numpy -- the same
    # place emager_proto.c does it on the device.
    embed_q, embed_f = run_int8_tflite(tflite, mav)
    if protos is not None:
        dev_logits = proto_logits(embed_f, protos)
    else:
        dev_logits = embed_f
    dev_class = dev_logits.argmax(1)
    ref_class = ref_logits.argmax(1)
    host_top1 = 100.0 * float((dev_class == ref_class).mean())
    logger.info(f"  INT8 vs FP32 PyTorch on these {len(pick)} windows: {host_top1:.1f}% top-1")
    if host_top1 < 100.0:
        logger.warning("  the INT8 model already disagrees with PyTorch on some of these "
                       "windows -- the device is being checked against the INT8 column, "
                       "so this is a ceiling, not a failure")

    out = Path(args.out) if args.out else (export_dir / "emager_test_vectors.h")
    write_test_vectors_header(out, mav, embed_q, dev_class, args.model, args.dataset,
                              args.tol_lsb, host_top1)
    logger.info(f"Done. Copy {out.name} into the firmware next to emager_model_params.h "
                f"and call emager_selftest_run().")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Emit on-device test vectors for an exported EMaGer model.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model class that was exported")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset folder under Datasets/")
    p.add_argument("--export-dir", default=None,
                   help="export directory (default: examples/deployment/exported/<model>)")
    p.add_argument("--checkpoint", default=None,
                   help="FP32 weights (default: <export-dir>/model_fp32.pt)")
    p.add_argument("--num", type=int, default=DEFAULT_NUM_VECTORS,
                   help=f"number of test vectors (default {DEFAULT_NUM_VECTORS})")
    p.add_argument("--tol-lsb", type=int, default=DEFAULT_TOL_LSB,
                   help=f"embedding tolerance in INT8 LSBs (default {DEFAULT_TOL_LSB})")
    p.add_argument("--out", default=None, help="output header path")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")
    run(args)


if __name__ == "__main__":
    main()
