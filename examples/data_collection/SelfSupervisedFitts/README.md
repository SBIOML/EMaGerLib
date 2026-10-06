# SelfSupervisedFitts

Self-supervised pretraining for EMG-based gesture recognition using SimCLR, with screen-guided
training (SGT) data collection for the EMaGer v3 armband using Eulerian gesture transitions.

Adapted from [eeddy/SelfSupervisedFitts](https://github.com/eeddy/SelfSupervisedFitts).

This folder is **self-contained**: it does not depend on `emagerlib`, and it uses its own
environment so it can be shared and run independently.

## Setup

Requires **Python 3.12** (tested on Windows) and **git** (to install libemg).

```bash
cd examples/data_collection/SelfSupervisedFitts

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python install_libemg_patch.py
```

`install_libemg_patch.py` is required. The project uses a pinned commit of libemg's
`transition-sgt` branch, plus local changes stored in `libemg_patch/`:

| File | Change |
|------|--------|
| `_streamers/_emagerv3_streamer.py` | EMaGer v3 serial streamer (`emg`, `imu`, `sample_id`) |
| `streamers.py` | Exposes `libemg.streamers.emagerv3_streamer()` |
| `_gui/_data_collection_panel.py` | Eulerian transition order (each `i -> j` transition exactly once) and a stream warm-up check |

Rerun the patch script if you reinstall libemg.

## Collecting data (EMaGer v3)

1. Plug in the EMaGer v3 armband.
2. Edit `config.py`: `CLASSES`, `NUM_REPS`, `REP_TIME`, and `SESSION` (the output folder name).
3. Run:

   ```bash
   python collect_data.py
   ```

Gesture images are downloaded automatically into `media-sgt/`. Each session is saved in
`<SESSION>/` next to `config.py`, as `C_<class>_R_<rep>_O_<order>_{emg,imu,sample_id}.csv`
plus `collection_details.json`.

To check a recorded session, run `python verify_emg.py` or `python verify_sample_id.py`. Both
read the `SESSION` set in `config.py`; adjust `N_FILES` at the bottom of each script.

`sequence_order.py` prints the Eulerian gesture order for a given number of classes.

## Data

Recorded sessions are **not** in git because of their size. Ask Félix Chamberland for the
recordings, then put each session folder directly next to `config.py`.

The original pretraining scripts expect the layout below:

```
data/
└── oymotion/
    ├── <subject_dirs>/
    │   └── ... *.csv files ...
    └── SGT/
        ├── train/
        └── test/
```

- All `.csv` files under the device directory (excluding `SGT/`) are discovered recursively and used for pretraining.
- The last 8 columns of each CSV are treated as EMG channels.
- `SGT/` contains labeled screen-guided training data used for evaluation during pretraining (train/test split with class and rep labels).
- A 90/10 train/validation split is applied to the discovered files.

## Training the backbone

```bash
python train_backbone.py
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | `trained_models` | Where to save the trained model and scaler |
| `--no-gru` | off | Disable the temporal GRU (encoder only) |

Outputs:

- `<output-dir>/backbone.pt`: trained encoder weights
- `<output-dir>/backbone_scaler.pkl`: fitted feature scaler
- `<output-dir>/checkpoints/`: periodic checkpoints during training

Then `python tune_model.py` calibrates the pretrained encoder on labeled SGT data.

## Architecture

- **Backbone** (`Backbone.py`): MLP encoder with LayerNorm, trained via SimCLR contrastive learning on WENG features extracted from augmented EMG windows.
- **Augmentations**: Gaussian noise, magnitude scaling, channel rotation, channel dropout, and temporal shift. They are applied at random to create contrastive pairs.
- **Calibration**: After pretraining, the encoder can be calibrated with labeled data using kNN, centroid classification, or a simple MLP (`MLP.py`).
- **Helpers** (`helpers/`): visualization of augmentations, data, and encoder embeddings.
