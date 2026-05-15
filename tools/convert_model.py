#!/usr/bin/env python3
"""
Convert EmagerCNN .pth models to portable formats.

Outputs:
  - ONNX     : universal format, works with ONNX Runtime and most edge frameworks
  - TFLite   : for TensorFlow Lite Micro on microcontrollers  (requires: pip install onnx2tf)
  - C header : TFLite bytes as a C uint8 array, ready to include in a Teensy/Arduino sketch

Usage:
  python tools/convert_model.py models/EM_3sessions.pth
  python tools/convert_model.py models/EM_3sessions.pth --tflite --c-array --guide
  python tools/convert_model.py models/EM_3sessions.pth --output-dir models/converted
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

# Windows cp1252 cannot encode torch's emoji-laden log messages
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _detect_params(state_dict: dict) -> tuple:
    """Infer (input_shape, num_classes) from state-dict tensor shapes."""
    # fc5.weight : [num_classes, 256]
    num_classes = state_dict["fc5.weight"].shape[0]
    # fc4.weight : [256, 32 * H * W]
    flat_size = state_dict["fc4.weight"].shape[1]
    spatial = flat_size // 32          # H * W  (= 64 for the standard 4×16 grid)
    rows = 4                           # EMaGer standard: 4 rows, 16 cols
    input_shape = (rows, spatial // rows)
    return input_shape, num_classes


def load_model(pth_path, input_shape=None, num_classes=None):
    from emagerlib.models.models import EmagerCNN

    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(pth_path, map_location="cpu")

    detected_shape, detected_classes = _detect_params(state_dict)
    if input_shape is None:
        input_shape = detected_shape
        print(f"  Auto-detected input_shape  = {input_shape}")
    if num_classes is None:
        num_classes = detected_classes
        print(f"  Auto-detected num_classes  = {num_classes}")

    model = EmagerCNN(input_shape, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, input_shape, num_classes


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def print_model_info(model, pth_path, input_shape):
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Leaf modules only (no containers)
    leaf_layers = [
        (name, type(m).__name__, sum(p.numel() for p in m.parameters()))
        for name, m in model.named_modules()
        if not list(m.children())
    ]
    type_counts: dict[str, int] = {}
    for _, mtype, _ in leaf_layers:
        type_counts[mtype] = type_counts.get(mtype, 0) + 1

    file_bytes = Path(pth_path).stat().st_size
    ram_f32 = total_params * 4  # bytes at float32

    W = 62
    print("\n" + "=" * W)
    print("  MODEL SUMMARY — EmagerCNN")
    print("=" * W)
    print(f"  File         : {pth_path}")
    print(f"  Input shape  : {input_shape}  (flat: {int(np.prod(input_shape))} features)")
    print(f"  Output       : {model.fc5.out_features} classes")
    print(f"  Parameters   : {total_params:,}  ({trainable:,} trainable)")
    print(f"  File size    : {file_bytes:,} B   ({file_bytes / 1024:.1f} KB)")
    print(f"  RAM (float32): {ram_f32:,} B   ({ram_f32 / 1024:.1f} KB)")
    print()
    print("  Leaf layers:")
    for name, mtype, nparams in leaf_layers:
        pstr = f"  [{nparams:,} params]" if nparams else ""
        print(f"    {name:<28} {mtype:<22}{pstr}")
    print()
    print("  Layer-type counts:")
    for mtype, cnt in sorted(type_counts.items()):
        print(f"    {mtype}: {cnt}")
    print("=" * W)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, input_shape, output_path: Path) -> Path:
    dummy = torch.randn(1, *input_shape)
    # dynamo=False uses the legacy TorchScript-based exporter, which is more
    # stable, doesn't require onnxscript, and works on Windows without emoji logging.
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["emg_features"],
        output_names=["class_logits"],
        dynamic_axes={
            "emg_features": {0: "batch"},
            "class_logits": {0: "batch"},
        },
        dynamo=False,
    )
    size = output_path.stat().st_size
    print(f"\n  [ONNX]   {output_path}")
    print(f"           {size:,} B  ({size / 1024:.1f} KB)")

    try:
        import onnx
        onnx.checker.check_model(str(output_path))
        print("           Graph check: OK")
    except ImportError:
        print("           (install 'onnx' to validate the graph)")

    return output_path


# ---------------------------------------------------------------------------
# TFLite export  (requires onnx2tf + tensorflow)
# ---------------------------------------------------------------------------

def export_tflite(onnx_path: Path, output_dir: Path):
    try:
        import onnx2tf
    except ImportError:
        print(
            "\n  [TFLITE] SKIP — onnx2tf not installed.\n"
            "           pip install onnx2tf\n"
            "           Then re-run with --tflite"
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [TFLITE] Converting {onnx_path.name} ...")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        non_verbose=True,
    )

    # onnx2tf names the file <stem>_float32.tflite
    candidates = sorted(output_dir.glob("*.tflite"))
    if not candidates:
        print("  [TFLITE] Conversion ran but no .tflite file found in output dir.")
        return None

    tflite_path = candidates[0]
    size = tflite_path.stat().st_size
    print(f"           {tflite_path}")
    print(f"           {size:,} B  ({size / 1024:.1f} KB)")
    return tflite_path


# ---------------------------------------------------------------------------
# TFLite → C array (for direct inclusion in Teensy/Arduino sketch)
# ---------------------------------------------------------------------------

def tflite_to_c_array(tflite_path: Path, output_path: Path, var_name: str) -> Path:
    data = tflite_path.read_bytes()
    hex_vals = [f"0x{b:02x}" for b in data]

    rows = []
    for i in range(0, len(hex_vals), 12):
        rows.append("  " + ", ".join(hex_vals[i : i + 12]) + ",")
    rows[-1] = rows[-1].rstrip(",")  # remove trailing comma

    content = "\n".join([
        "// Auto-generated by tools/convert_model.py — do not edit manually.",
        f"// Source : {tflite_path.name}",
        f"// Size   : {len(data)} bytes",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"alignas(8) const uint8_t {var_name}[] = {{",
        *rows,
        "};",
        f"const unsigned int {var_name}_len = {len(data)};",
        "",
    ])

    output_path.write_text(content, encoding="utf-8")
    print(f"\n  [C HDR]  {output_path}")
    print(f"           {len(data):,} B  ({len(data) / 1024:.1f} KB)  var: {var_name}")
    return output_path


# ---------------------------------------------------------------------------
# Teensy integration guide
# ---------------------------------------------------------------------------

def print_teensy_guide(c_header_path, num_classes):
    header = c_header_path.name if c_header_path else "model_data.h"
    var = c_header_path.stem if c_header_path else "emager_model"

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   TEENSY 4.1 — TFLITE MICRO INFERENCE GUIDE             ║
╚══════════════════════════════════════════════════════════╝

STEP 1 — Install EloquentTinyML
  Arduino Library Manager → search "EloquentTinyML" (Simone Salerno) → v2+
  (Internally wraps TensorFlow Lite Micro; works on Teensy 4.x)

STEP 2 — Copy "{header}" into your sketch folder.

STEP 3 — Arduino sketch template
─────────────────────────────────────────────────────────────
#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include "{header}"

// 64 MAV features in, {num_classes} gesture-class scores out
#define N_INPUTS   64
#define N_OUTPUTS  {num_classes}
#define ARENA_SIZE 30000   // bytes — increase if ml.isOk() fails

Eloquent::TinyML::TensorFlow::TensorFlow<N_INPUTS, N_OUTPUTS, ARENA_SIZE> ml;

void setup() {{
  Serial.begin(115200);
  if (!ml.begin({var}))
    Serial.println(ml.errorMessage());
}}

void loop() {{
  // ── Compute MAV features from one EMG window ──────────────
  // Window: WINDOW_SIZE samples × 64 channels
  // features[ch] = mean( |window[ch][0..WINDOW_SIZE-1]| )
  float features[N_INPUTS];
  // fill features[] here ...

  // ── Run inference ─────────────────────────────────────────
  float scores[N_OUTPUTS];
  int gesture = ml.predict(features, scores);

  Serial.print("Gesture: ");  Serial.print(gesture);
  Serial.print("  scores: ");
  for (int i = 0; i < N_OUTPUTS; i++) {{
    Serial.print(scores[i], 3);  Serial.print(' ');
  }}
  Serial.println();
  delay(10);
}}
─────────────────────────────────────────────────────────────

NOTES
  • ARENA_SIZE: start at 30 000. Teensy 4.1 has 1 MB RAM so headroom is large.
  • Input order: the 64 floats must match channel order used during training
    (MAV, no z-score normalization — BatchNorm is baked into the model weights).
  • Class index mapping: same as CLASSES list in your config
    e.g. CLASSES=[1,2,3,14,18,19,30] → index 0=gesture 1, 1=gesture 2, …
  • Latency: Teensy 4.1 (Cortex-M7 @ 600 MHz + FPU) typically runs this
    model in < 1 ms per inference.
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pth", help="Path to .pth model file")
    parser.add_argument(
        "--input-shape", nargs=2, type=int, metavar=("H", "W"), default=None,
        help="Input grid shape (default: auto-detect, usually 4 16)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of output classes (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same folder as .pth)",
    )
    parser.add_argument(
        "--no-onnx", action="store_true",
        help="Skip ONNX export",
    )
    parser.add_argument(
        "--tflite", action="store_true",
        help="Convert to TFLite  (requires: pip install onnx2tf tensorflow)",
    )
    parser.add_argument(
        "--c-array", action="store_true",
        help="Embed TFLite model as C header for Teensy  (implies --tflite)",
    )
    parser.add_argument(
        "--guide", action="store_true",
        help="Print Teensy/Arduino integration guide",
    )
    args = parser.parse_args(argv)

    pth = Path(args.pth).resolve()
    if not pth.exists():
        sys.exit(f"Error: file not found: {pth}")

    out_dir = (args.output_dir or pth.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_shape = tuple(args.input_shape) if args.input_shape else None

    print(f"\nLoading {pth.name} …")
    model, input_shape, num_classes = load_model(pth, input_shape, args.num_classes)

    print_model_info(model, pth, input_shape)

    onnx_path = tflite_path = c_path = None

    if not args.no_onnx:
        onnx_path = out_dir / (pth.stem + ".onnx")
        export_onnx(model, input_shape, onnx_path)

    if args.tflite or args.c_array:
        if onnx_path is None:
            onnx_path = out_dir / (pth.stem + ".onnx")
            export_onnx(model, input_shape, onnx_path)
        tflite_path = export_tflite(onnx_path, out_dir / "tflite")

    if args.c_array:
        if tflite_path is None:
            sys.exit("Error: TFLite conversion failed — cannot generate C array.")
        var_name = pth.stem.replace("-", "_") + "_model"
        c_path = out_dir / (pth.stem + "_model.h")
        tflite_to_c_array(tflite_path, c_path, var_name)

    if args.guide or args.c_array:
        print_teensy_guide(c_path, num_classes)

    print("\nDone.")


if __name__ == "__main__":
    main()
