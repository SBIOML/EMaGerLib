import time
import os
import re
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
import logging

import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg

from libemg.gui import GUI
from libemg._gui._data_collection_panel import DataCollectionPanel
from libemg._gui._utils import set_texture

from emagerlib.config.load_config import load_config
from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.utils.streamer_utils import get_emager_streamer

# Default configuration path
DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py"

# Default limb position images folder (lp0.png, lp1.png, ...). Can be overridden with cfg.LIMB_POSITIONS_PATH
DEFAULT_LIMB_POSITIONS_PATH = Path(r"C:\Users\felix\OneDrive\Documents\phD\Projet\Data_collection_paper2\limb_positions")

# Data files written by libemg: C_<class>_R_<rep>_<modality>.csv
DATA_FILE_PATTERN = re.compile(r"^C_(\d+)_R_(\d+)_(.+)\.csv$")

# Create module logger (inherits from root logger configured above)
logger = logging.getLogger(__name__)


def load_limb_textures(limb_folder, width, height):
    """Load lp*.png files, letterboxed on a white background to (width, height) without distortion.
    Returns a list of (color_texture, grayscale_texture), index i corresponds to lp{i}."""
    files = sorted(Path(limb_folder).glob("lp*.png"), key=lambda f: int(f.stem[2:]))
    if not files:
        raise FileNotFoundError(f"No limb position images (lp*.png) found in '{limb_folder}'.")

    textures = []
    for f in files:
        img = Image.open(f).convert("RGBA")
        scale = min(width / img.width, height / img.height)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2), img)
        color = np.asarray(canvas, dtype=np.float32).ravel() / 255.0
        gray = np.asarray(canvas.convert("L").convert("RGBA"), dtype=np.float32).ravel() / 255.0
        textures.append((color, gray))
    logger.info(f"Loaded {len(textures)} limb positions from '{limb_folder}'")
    return textures


def list_gesture_files(media_folder):
    """Gesture media files, in the same order libemg's DataCollectionPanel.gather_media uses for class indices."""
    files = sorted(os.listdir(media_folder))
    return [file for file in files if file.endswith((".gif", ".png", ".mp4", "jpg"))]


def find_resume_point(data_folder, gesture_files, num_reps, modalities):
    """Find the first repetition that is not fully recorded.

    A rep is complete when every gesture has a non-empty file for every modality.
    Returns (start_rep, files_of_incomplete_rep), or (None, []) if the saved classes
    do not match the current media folder (resuming would mix up class indices)."""
    details_path = Path(data_folder, "collection_details.json")
    classes = [f.split('.')[0] for f in gesture_files]
    if details_path.exists():
        with open(details_path, 'r') as f:
            saved_classes = json.load(f).get("classes")
        if saved_classes is not None and saved_classes != classes:
            logger.warning(f"Saved classes {saved_classes} do not match the media folder {classes}. Cannot resume.")
            return None, []

    recorded = {}  # rep -> {(class_index, modality)}
    data_files = []
    for file in os.listdir(data_folder):
        match = DATA_FILE_PATTERN.match(file)
        if not match:
            continue
        data_files.append((file, int(match[2])))
        if os.path.getsize(Path(data_folder, file)) > 0:
            recorded.setdefault(int(match[2]), set()).add((int(match[1]), match[3]))

    expected = {(c, mod) for c in range(len(gesture_files)) for mod in modalities}
    start_rep = 0
    while start_rep < num_reps and expected <= recorded.get(start_rep, set()):
        start_rep += 1

    later_reps = sorted({rep for _, rep in data_files if rep > start_rep})
    if later_reps:
        logger.warning(f"Found data for reps {later_reps} after incomplete rep {start_rep}; they will be overwritten when reached.")
    return start_rep, [file for file, rep in data_files if rep == start_rep]


class LimbPositionDataCollectionPanel(DataCollectionPanel):
    """Same as libemg's DataCollectionPanel, but shows the limb position to the right of the gesture.
    The limb position cycles with the repetition index: rep r -> lp(r % n_limb_positions).
    If start_rep > 0, the collection resumes at that repetition."""

    def __init__(self, *args, limb_textures=None, limb_width=250, start_rep=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.limb_textures = limb_textures
        self.limb_width = limb_width
        self.limb_height = self.video_player_height
        self.start_rep = start_rep

    def limb_index(self, rep_index):
        return rep_index % len(self.limb_textures)

    def gather_media(self):
        details_path = Path(self.output_folder, "collection_details.json").absolute().as_posix()
        previous_details = {}
        if self.start_rep > 0 and os.path.exists(details_path):
            with open(details_path, 'r') as f:
                previous_details = json.load(f)

        collection_conf = super().gather_media()

        with open(details_path, 'r') as f:
            collection_details = json.load(f)
        # Store the rep -> limb position mapping alongside the collection details
        collection_details["limb_positions"] = {rep: f"lp{self.limb_index(rep)}" for rep in range(self.num_reps)}
        if self.start_rep > 0:
            # Keep the original start time and log when the session was resumed
            collection_details["time"] = previous_details.get("time", collection_details["time"])
            collection_details["resumed"] = previous_details.get("resumed", []) + [
                {"time": datetime.now().isoformat(), "start_rep": self.start_rep}]
        with open(details_path, 'w') as f:
            json.dump(collection_details, f)

        if self.start_rep >= self.num_reps:
            raise ValueError(f"Resuming at rep {self.start_rep + 1}, but Num Reps is {self.num_reps}. Nothing left to record.")
        # Skip the reps that were already recorded
        return [media for media in collection_conf if media[3] >= self.start_rep]

    def set_limb_texture(self, rep_index, active=True):
        color, gray = self.limb_textures[self.limb_index(rep_index)]
        set_texture("__dc_limb_visual", color if active else gray, self.limb_width, self.limb_height)

    def spawn_collection_window(self, media_list):
        # open first frame of gif
        self.gui.online_data_handler.prepare_smm()
        texture = media_list[0][0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height)
        set_texture("__dc_collection_visual", texture, width=self.video_player_width, height=self.video_player_height)
        self.set_limb_texture(media_list[0][3])

        collection_window_width  = self.video_player_width + self.limb_width + 130
        collection_window_height = self.video_player_height + 300
        with dpg.window(label="Collection Window",
                        tag="__dc_collection_window",
                        width=collection_window_width,
                        height=collection_window_height):

            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=self.video_player_width/2+30-(7*len("Collection Menu"))/2)
                dpg.add_text(default_value="Collection Menu")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_rep_spacer",height=20,width=self.video_player_width/2+30 - (7*len(media_list[0][1]))/2)
                dpg.add_text(f"Rep {media_list[0][3] + 1} of {self.num_reps}", tag="__dc_rep")
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_prompt_spacer",height=20,width=self.video_player_width/2+30 - (7*len(media_list[0][1]))/2)
                dpg.add_text(media_list[0][1], tag="__dc_prompt")
            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=30)
                dpg.add_image("__dc_collection_visual")
                dpg.add_spacer(width=20)
                dpg.add_image("__dc_limb_visual")
            with dpg.group(horizontal=True):
                dpg.add_spacer(height=20,width=30)
                dpg.add_progress_bar(tag="__dc_progress", default_value=0.0,width=self.video_player_width)
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_stop_spacer", height=20, width=self.video_player_width/2+30 - (7*len("STOP REP"))/2)
                dpg.add_button(tag="__dc_stop_button", label="STOP REP", callback=self.stop_rep_callback)
                dpg.bind_item_theme("__dc_stop_button", self.get_stop_button_theme())
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_redo_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Redo"))/2)
                dpg.add_button(tag="__dc_redo_button", label="Redo", callback=self.redo_collection_callback)
            with dpg.group(horizontal=True):
                dpg.add_spacer(tag="__dc_continue_spacer", height=20, width=self.video_player_width/2+30 - (7*len("Continue"))/2)
                dpg.add_button(tag="__dc_continue_button", label="Continue", callback=self.continue_collection_callback)
            with dpg.group(horizontal=True, tag="__dc_rest_group"):
                dpg.add_spacer(height=20, width=self.video_player_width/2+30 - (7*len("Rest between gestures (s): ")+120)/2)
                dpg.add_text("Rest between gestures (s): ")
                dpg.add_input_float(tag="__dc_rest_input", default_value=float(self.rest_time), width=120,
                                    step=0.5, min_value=0.0, min_clamped=True, format="%.1f")
            dpg.hide_item(item="__dc_redo_button")
            dpg.hide_item(item="__dc_continue_button")
            dpg.hide_item(item="__dc_stop_button")
            dpg.hide_item(item="__dc_rest_group")

        self.run_sgt(media_list)
        # clean up the window

        dpg.delete_item("__dc_collection_window")
        self.cleanup_window("collection")
        # Session finished: a new Start from the menu records from rep 0 again
        self.start_rep = 0
        # open config back up
        self.spawn_configuration_window()

    def get_stop_button_theme(self):
        if not dpg.does_item_exist("__dc_stop_theme"):
            with dpg.theme(tag="__dc_stop_theme"):
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 30, 30))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 45, 45))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (240, 60, 60))
        return "__dc_stop_theme"

    def run_sgt(self, media_list):
        # Same as libemg's run_sgt, plus the emergency stop (STOP REP button)
        self.i = 0
        self.advance = True
        self.stop_requested = False
        self.gui.online_data_handler.reset()
        while self.i < len(media_list):
            self.rep_buffer = {mod:[] for mod in self.gui.online_data_handler.modalities}
            self.rep_count  = {mod:0 for mod in self.gui.online_data_handler.modalities}
            dpg.show_item(item="__dc_stop_button")
            # do the rest
            if self.rest_time and self.i < len(media_list):
                self.play_collection_visual(media_list[self.i], active=False)
                media_list[self.i][0].reset()
            self.gui.online_data_handler.reset()

            if not self.stop_requested:
                self.play_collection_visual(media_list[self.i], active=True)
            dpg.hide_item(item="__dc_stop_button")

            if self.stop_requested:
                # Discard the current rep and wait before starting it over
                rep = self.restart_current_rep(media_list)
                self.wait_for_continue(show_redo=False)
                dpg.set_value('__dc_rep', value=f"Rep {rep + 1} of {self.num_reps}")
                continue

            output_path = Path(self.output_folder, "C_" + str(media_list[self.i][2]) + "_R_" + str(media_list[self.i][3]) + ".csv").absolute().as_posix()
            self.save_data(output_path)
            last_rep = media_list[self.i][3]
            self.i = self.i+1
            is_final_media = self.i == len(media_list)
            if is_final_media:
                # At the end of the list, so we must be finished a rep
                rep_is_finished = True
                current_rep = self.num_reps - 1
            else:
                # Check if we've finished a rep
                current_rep = media_list[self.i][3]
                rep_is_finished = last_rep != current_rep

            # pause / redo goes here!
            if rep_is_finished  or (not self.auto_advance):
                self.wait_for_continue(show_redo=True)
                if not is_final_media:
                    dpg.set_value('__dc_rep', value=f"Rep {media_list[self.i][3] + 1} of {self.num_reps}")

    def wait_for_continue(self, show_redo=True):
        # Show redo / continue buttons and wait until one is pressed
        self.advance = False
        if show_redo:
            dpg.show_item(item="__dc_redo_button")
        dpg.show_item(item="__dc_continue_button")
        dpg.set_value("__dc_rest_input", float(self.rest_time))
        dpg.show_item(item="__dc_rest_group")
        while not self.advance:
            time.sleep(0.1)
            dpg.configure_app(manual_callback_management=True)
            jobs = dpg.get_callback_queue()
            dpg.run_callbacks(jobs)
        dpg.configure_app(manual_callback_management=False)
        dpg.hide_item(item="__dc_rest_group")
        self.update_rest_time(dpg.get_value("__dc_rest_input"))

    def update_rest_time(self, value):
        # Rest (not recorded) before each gesture, editable between reps
        try:
            new_rest_time = max(0.0, float(value))
        except (TypeError, ValueError):
            return
        if new_rest_time != self.rest_time:
            logger.info(f"Rest between gestures changed from {self.rest_time} s to {new_rest_time} s.")
            self.rest_time = new_rest_time

    def stop_rep_callback(self):
        self.stop_requested = True
        dpg.hide_item(item="__dc_stop_button")

    def restart_current_rep(self, media_list):
        """Move back to the first gesture of the current rep and delete the files already saved for it."""
        rep = media_list[self.i][3]
        while self.i > 0 and media_list[self.i - 1][3] == rep:
            self.i -= 1
        for file in os.listdir(self.output_folder):
            match = DATA_FILE_PATTERN.match(file)
            if match and int(match[2]) == rep:
                os.remove(os.path.join(self.output_folder, file))
                logger.info(f"Emergency stop: deleted '{file}'.")
        for media in media_list:
            if media[3] == rep:
                media[0].reset()
        self.stop_requested = False
        logger.info(f"Emergency stop: rep {rep + 1} discarded, waiting to start it over.")

        message = f"Rep {rep + 1} stopped. Press Continue to start it over."
        dpg.set_value("__dc_prompt", value=message)
        dpg.set_item_width("__dc_prompt_spacer", width=self.video_player_width/2+30 - (7*len(message))/2)
        dpg.set_value("__dc_progress", value=0.0)
        return rep

    def play_collection_visual(self, media, active=True):
        # Same as libemg's play_collection_visual, plus the limb position and an early exit on emergency stop
        # media = [Media, class_name, class_index, rep_index, rep_time]
        self.set_limb_texture(media[3], active=active)
        if active:
            timer_duration = media[-1]
            dpg.set_value("__dc_prompt", value=media[1])
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len(media[1]))/2)
        else:
            timer_duration = self.rest_time
            dpg.set_value("__dc_prompt", value="Up next: "+media[1])
            dpg.set_item_width("__dc_prompt_spacer",width=self.video_player_width/2+30 - (7*len("Up next: "+media[1]))/2)

        texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
        set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
        # initialize motion and frame timers
        motion_timer = time.perf_counter_ns()
        while (time.perf_counter_ns() - motion_timer)/1e9 < timer_duration:
            if self.stop_requested:
                return
            time.sleep(1/media[0].fps) # never refresh faster than media fps
            # update visual
            media[0].advance_to((time.perf_counter_ns() - motion_timer)/1e9)
            texture = media[0].get_dpg_formatted_texture(width=self.video_player_width,height=self.video_player_height, grayscale=not(active))
            set_texture("__dc_collection_visual", texture, self.video_player_width, self.video_player_height)
            # update progress bar
            progress = min(1,(time.perf_counter_ns() - motion_timer)/(1e9*timer_duration))
            # grab incoming new data
            if active:
                vals, count = self.gui.online_data_handler.get_data()
                for mod in self.gui.online_data_handler.modalities:
                    new_samples = count[mod][0][0]-self.rep_count[mod]
                    self.rep_buffer[mod] = [vals[mod][:new_samples,:]] + self.rep_buffer[mod]
                    self.rep_count[mod]  = self.rep_count[mod] + new_samples

            dpg.set_value("__dc_progress", value = progress)


class LimbPositionGUI(GUI):
    """libemg GUI that uses LimbPositionDataCollectionPanel for data collection."""

    def __init__(self, *args, limb_textures=None, limb_width=250, start_rep=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.limb_textures = limb_textures
        self.limb_width = limb_width
        self.start_rep = start_rep

    def _data_collection_callback(self):
        import inspect
        panel_arguments = list(inspect.signature(DataCollectionPanel.__init__).parameters)
        passed_arguments = {i: self.args[i] for i in self.args.keys() if i in panel_arguments}
        self.dcp = LimbPositionDataCollectionPanel(**passed_arguments, gui=self,
                                                   video_player_width=self.video_player_width,
                                                   video_player_height=self.video_player_height,
                                                   limb_textures=self.limb_textures,
                                                   limb_width=self.limb_width,
                                                   start_rep=self.start_rep)
        self.dcp.spawn_configuration_window()


def parse_args(argv=None):
    parser = create_parser(
        description="Screen-guided EMG training data collection with limb positions",
        default_config=str(DEFAULT_CONFIG)
    )
    return parser.parse_args(argv)


def setup_runtime(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(args, cfg, script_name="SGT_limbposition")
    save_config_if_requested(args, cfg, script_name="SGT_limbposition")
    return args, cfg


def main(argv=None):
    from libemg.data_handler import OnlineDataHandler

    _, cfg = setup_runtime(argv)
    media_folder = os.path.join(str(cfg.MEDIA_PATH), "")
    # cfg.DATAFOLDER is a read-only property, so the (possibly renamed) output folder is tracked here
    data_folder = str(cfg.DATAFOLDER)

    # Load limb position images
    limb_folder = getattr(cfg, "LIMB_POSITIONS_PATH", None) or DEFAULT_LIMB_POSITIONS_PATH
    limb_width = 250
    gesture_height = 500  # libemg GUI default gesture size (500x500)
    limb_textures = load_limb_textures(limb_folder, limb_width, gesture_height)

    # Create data handler and streamer
    p, smi = get_emager_streamer(cfg.EMAGER_VERSION)
    logger.info(f"Streamer created: process: {p}, smi : {smi}")
    odh = OnlineDataHandler(shared_memory_items=smi)
    logger.info("Data handler created")

    gui_args = {
        "online_data_handler": odh,
        "media_folder": media_folder,
        "data_folder": data_folder,
        "num_reps": cfg.NUM_REPS,
        "rep_time": cfg.REP_TIME,
        "rest_time": cfg.REST_TIME,
        "auto_advance": True
    }

    gui = LimbPositionGUI(odh, args=gui_args, debug=False, width=1000, height=800,
                          gesture_width=500, gesture_height=gesture_height,
                          limb_textures=limb_textures, limb_width=limb_width)
    gui.download_gestures(cfg.CLASSES, media_folder, download_gifs=False)

    dataset_path = os.path.abspath(data_folder)

    # Ensure folder exists
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    # Check if folder has any files/subfolders
    has_data = any(os.scandir(dataset_path))

    if has_data:
        logger.info(f"Data found in '{dataset_path}'.")

        # Find where the previous run stopped
        start_rep, incomplete_files = find_resume_point(dataset_path, list_gesture_files(media_folder),
                                                        cfg.NUM_REPS, odh.modalities)
        can_resume = start_rep is not None and start_rep < cfg.NUM_REPS
        if can_resume:
            logger.info(f"Reps 1 to {start_rep} of {cfg.NUM_REPS} are complete. "
                        f"Resuming at rep {start_rep + 1} (limb position lp{start_rep % len(limb_textures)}).")
            if incomplete_files:
                logger.info(f"The {len(incomplete_files)} files of incomplete rep {start_rep + 1} will be deleted.")
            prompt = "Choose action - [Enter] Resume, [O]verwrite, [R]ename, [C]ancel: "
        else:
            if start_rep is not None:
                logger.info(f"All {cfg.NUM_REPS} reps are already recorded in this session.")
            prompt = "Choose action - [O]verwrite, [R]ename, [C]ancel: "

        while True:
            choice = input(prompt).strip().lower()
            if can_resume and choice in ("", "resume"):
                for file in incomplete_files:
                    os.remove(os.path.join(dataset_path, file))
                    logger.info(f"Deleted incomplete file '{file}'.")
                gui.start_rep = start_rep
                break
            elif choice == "o" or choice == "overwrite":
                confirm = input(f"Are you sure you want to DELETE all contents of '{dataset_path}'? Type 'yes' to confirm: ").strip().lower()
                if confirm == "yes":
                    shutil.rmtree(dataset_path)
                    os.makedirs(dataset_path, exist_ok=True)
                    logger.info(f"Contents of '{dataset_path}' deleted.")
                    break
                else:
                    logger.info("Overwrite cancelled.")
                    continue
            elif choice == "r" or choice == "rename":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_path = f"{dataset_path}_{timestamp}"
                # Ensure unique name
                i = 1
                candidate = new_path
                while os.path.exists(candidate):
                    candidate = f"{new_path}_{i}"
                    i += 1
                os.makedirs(candidate, exist_ok=True)
                data_folder = candidate
                gui_args["data_folder"] = data_folder
                logger.info(f"Dataset directory renamed to '{data_folder}'. Using that path for this run.")
                break
            elif choice == "c" or choice == "cancel":
                logger.info("Operation cancelled. Exiting.")
                sys.exit(0)
            else:
                logger.warning(f"Invalid choice. {prompt}")
    else:
        # Ensure folder exists (redundant safe-guard)
        os.makedirs(dataset_path, exist_ok=True)

    gui.start_gui()

    logger.info(f"Data saved in: {data_folder}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
