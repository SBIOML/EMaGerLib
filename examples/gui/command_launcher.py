"""GUI command launcher for EMaGerLib."""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
from importlib import import_module
from typing import List, Optional, Union

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # pragma: no cover
    tk = None
    ttk = None


DARK_BG = "#1E1E1E"
DARK_PANEL = "#252526"
DARK_BORDER = "#3C3C3C"
DARK_INPUT = "#2D2D30"
DARK_BUTTON = "#333337"
DARK_BUTTON_ACTIVE = "#3F3F46"
DARK_TEXT = "#E6E6E6"
ACCENT_TEXT = "#9CDCFE"


def split_arguments(argument_string: str) -> List[str]:
    """Split user-provided argument text into an argv list."""
    if not argument_string.strip():
        return []

    use_posix = os.name != "nt"
    return shlex.split(argument_string, posix=use_posix)


def get_available_emager_commands() -> List[str]:
    """Return available unified CLI commands, excluding the GUI command itself."""
    main_module = import_module("examples.main")
    command_map = getattr(main_module, "COMMANDS", {})
    return sorted(command for command in command_map.keys() if command != "gui")


def build_emager_command(command_name: str, argument_string: str = "") -> List[str]:
    """Build a Python command line that runs the selected unified emager command."""
    return [sys.executable, "-m", "examples.main", command_name, *split_arguments(argument_string)]


class CommandLauncherGUI:
    """Desktop launcher for EMaGerLib commands."""

    def __init__(self) -> None:
        if tk is None or ttk is None:  # pragma: no cover
            raise RuntimeError("Tkinter is not available in this Python environment.")

        self.root = tk.Tk()
        self.root.title("EMaGerLib Command Launcher")
        self.root.geometry("980x640")
        self._apply_dark_theme()

        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.command_var = tk.StringVar(value="train-cnn")
        self.args_var = tk.StringVar()
        self.custom_command_var = tk.StringVar()

        self._build_layout()
        self._populate_presets()
        self._poll_output_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_dark_theme(self) -> None:
        self.root.configure(bg=DARK_BG)

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=DARK_BG)
        style.configure(
            "TLabelframe",
            background=DARK_BG,
            foreground=DARK_TEXT,
            bordercolor=DARK_BORDER,
            lightcolor=DARK_BORDER,
            darkcolor=DARK_BORDER,
        )
        style.configure("TLabelframe.Label", background=DARK_BG, foreground=ACCENT_TEXT)
        style.configure("TLabel", background=DARK_BG, foreground=DARK_TEXT)
        style.configure(
            "TButton",
            background=DARK_BUTTON,
            foreground=DARK_TEXT,
            bordercolor=DARK_BORDER,
            focusthickness=1,
            focuscolor=DARK_BORDER,
            padding=6,
        )
        style.map(
            "TButton",
            background=[("active", DARK_BUTTON_ACTIVE), ("pressed", DARK_BUTTON_ACTIVE)],
            foreground=[("disabled", "#7A7A7A")],
        )
        style.configure(
            "TEntry",
            fieldbackground=DARK_INPUT,
            foreground=DARK_TEXT,
            insertcolor=DARK_TEXT,
            bordercolor=DARK_BORDER,
        )
        style.configure(
            "TCombobox",
            fieldbackground=DARK_INPUT,
            background=DARK_INPUT,
            foreground=DARK_TEXT,
            arrowcolor=DARK_TEXT,
            bordercolor=DARK_BORDER,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", DARK_INPUT)],
            foreground=[("readonly", DARK_TEXT)],
            selectbackground=[("readonly", DARK_INPUT)],
            selectforeground=[("readonly", DARK_TEXT)],
        )

        self.root.option_add("*TCombobox*Listbox.background", DARK_INPUT)
        self.root.option_add("*TCombobox*Listbox.foreground", DARK_TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", DARK_BUTTON_ACTIVE)
        self.root.option_add("*TCombobox*Listbox.selectForeground", DARK_TEXT)

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        preset_group = ttk.LabelFrame(main_frame, text="Run EMaGer Command", padding=10)
        preset_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(preset_group, text="Command:").grid(row=0, column=0, sticky="w")
        self.command_combobox = ttk.Combobox(
            preset_group,
            textvariable=self.command_var,
            state="readonly",
            width=38,
        )
        self.command_combobox.grid(row=0, column=1, sticky="we", padx=8)

        ttk.Label(preset_group, text="Extra args:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(preset_group, textvariable=self.args_var, width=60).grid(
            row=1,
            column=1,
            sticky="we",
            padx=8,
            pady=(8, 0),
        )

        ttk.Button(preset_group, text="Run Selected", command=self._run_selected_command).grid(
            row=0,
            column=2,
            rowspan=2,
            padx=(10, 0),
            sticky="ns",
        )

        preset_group.columnconfigure(1, weight=1)

        custom_group = ttk.LabelFrame(main_frame, text="Run Custom Shell Command", padding=10)
        custom_group.pack(fill=tk.X, padx=4, pady=8)

        ttk.Entry(custom_group, textvariable=self.custom_command_var).grid(
            row=0,
            column=0,
            sticky="we",
        )
        ttk.Button(custom_group, text="Run Custom", command=self._run_custom_command).grid(
            row=0,
            column=1,
            padx=(10, 0),
        )
        custom_group.columnconfigure(0, weight=1)

        controls_group = ttk.Frame(main_frame)
        controls_group.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(controls_group, text="Stop Running", command=self._stop_running_process).pack(side=tk.LEFT)
        ttk.Button(controls_group, text="Clear Output", command=self._clear_output).pack(side=tk.LEFT, padx=8)

        self.status_label = ttk.Label(controls_group, text="Idle")
        self.status_label.pack(side=tk.RIGHT)

        output_group = ttk.LabelFrame(main_frame, text="Output", padding=8)
        output_group.pack(fill=tk.BOTH, expand=True, padx=4, pady=(8, 4))

        self.output_text = tk.Text(
            output_group,
            wrap=tk.WORD,
            height=20,
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            insertbackground=DARK_TEXT,
            selectbackground="#264F78",
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll = ttk.Scrollbar(output_group, orient=tk.VERTICAL, command=self.output_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.configure(yscrollcommand=scroll.set)

    def _populate_presets(self) -> None:
        commands = get_available_emager_commands()
        self.command_combobox["values"] = commands
        if commands:
            self.command_var.set(commands[0])

    def _append_output(self, text: str) -> None:
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def _clear_output(self) -> None:
        self.output_text.delete("1.0", tk.END)

    def _set_status(self, status_text: str) -> None:
        self.status_label.config(text=status_text)

    def _run_selected_command(self) -> None:
        selected_command = self.command_var.get().strip()
        if not selected_command:
            self._append_output("Please select a command.\n")
            return

        command_line = build_emager_command(selected_command, self.args_var.get())
        self._start_process(command_line, shell=False)

    def _run_custom_command(self) -> None:
        custom_command = self.custom_command_var.get().strip()
        if not custom_command:
            self._append_output("Please provide a custom command.\n")
            return

        self._start_process(custom_command, shell=True)

    def _start_process(self, command_line: Union[str, List[str]], shell: bool) -> None:
        if self.process is not None and self.process.poll() is None:
            self._append_output("A process is already running. Stop it before starting another one.\n")
            return

        self._append_output(f"\n$ {command_line}\n")
        self._set_status("Running")

        try:
            self.process = subprocess.Popen(
                command_line,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                shell=shell,
            )
        except Exception as exc:
            self.process = None
            self._set_status("Error")
            self._append_output(f"Failed to start command: {exc}\n")
            return

        worker = threading.Thread(target=self._stream_output, daemon=True)
        worker.start()

    def _stream_output(self) -> None:
        current_process = self.process
        if current_process is None:
            return

        if current_process.stdout is not None:
            for line in current_process.stdout:
                self.output_queue.put(line)

        return_code = current_process.wait()
        self.output_queue.put(f"\nProcess finished with exit code {return_code}.\n")
        self.process = None

    def _poll_output_queue(self) -> None:
        drained = False
        while True:
            try:
                line = self.output_queue.get_nowait()
            except queue.Empty:
                break

            drained = True
            self._append_output(line)

        if drained and (self.process is None or self.process.poll() is not None):
            self._set_status("Idle")

        self.root.after(100, self._poll_output_queue)

    def _stop_running_process(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self._append_output("No running process to stop.\n")
            return

        self.process.terminate()
        self._append_output("Stop signal sent.\n")
        self._set_status("Stopping")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    """Entry point for the GUI launcher."""
    if tk is None:
        raise RuntimeError("Tkinter is required for the GUI launcher but is not installed.")

    app = CommandLauncherGUI()
    app.run()


if __name__ == "__main__":
    main()
