"""
Copy the patched libemg files from libemg_patch/ into the installed libemg package.

The patch adds:
  - _streamers/_emagerv3_streamer.py : EMaGer v3 serial streamer (emg, imu, sample_id)
  - streamers.py                     : exposes libemg.streamers.emagerv3_streamer()
  - _gui/_data_collection_panel.py   : Eulerian transition order + stream warm-up gate

Run once, after `pip install -r requirements.txt`, inside the same environment.
"""
import importlib.util
import os
import shutil

PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libemg_patch")


def main():
    spec = importlib.util.find_spec("libemg")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("libemg is not installed. Run `pip install -r requirements.txt` first.")
    libemg_dir = list(spec.submodule_search_locations)[0]

    for root, _, files in os.walk(PATCH_DIR):
        for name in files:
            if not name.endswith(".py"):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(libemg_dir, os.path.relpath(src, PATCH_DIR))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"patched {os.path.relpath(dst, libemg_dir)}")

    print(f"libemg patched in {libemg_dir}")


if __name__ == "__main__":
    main()
