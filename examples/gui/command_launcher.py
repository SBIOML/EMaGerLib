"""PyQt-based GUI command launcher for EMaGerLib (comparison implementation)."""

import ast
import os
import shlex
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

try:
    from PyQt6.QtCore import QProcess
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QComboBox,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QScrollArea,
    )
except ImportError:  # pragma: no cover
    QProcess = None
    QApplication = None

from emagerlib.config.core_config import CoreConfig
from emagerlib.config.load_config import load_config
from emagerlib.config.save_config import save_config


def get_available_emager_commands():
    """Return available emager commands excluding the GUI launcher itself."""
    from examples.main import COMMANDS

    return sorted(command for command in COMMANDS if command != "gui")


def build_emager_command(command_name, extra_args=""):
    """Build a command list to execute an emager subcommand via Python module entry."""
    command = [sys.executable, "-m", "examples.main", command_name]
    if extra_args and extra_args.strip():
        command.extend(shlex.split(extra_args, posix=(os.name != "nt")))
    return command


LIGHT_STYLESHEET = """
QWidget {
    background-color: #f4f5f7;
    color: #202124;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #d0d3d8;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #005a9e;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #ffffff;
    color: #202124;
    border: 1px solid #c7ccd3;
    border-radius: 6px;
    padding: 6px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #202124;
    selection-background-color: #dbeeff;
}
QPushButton {
    background-color: #e9edf2;
    color: #202124;
    border: 1px solid #c7ccd3;
    border-radius: 8px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #dde3ea;
}
QPushButton#themeButton {
    min-width: 30px;
    max-width: 30px;
    padding: 4px;
    border-radius: 15px;
    font-size: 14px;
    color: #005a9e;
}
"""


DARK_STYLESHEET = """
QWidget {
    background-color: #1f2125;
    color: #e8eaed;
    font-size: 10pt;
}
QGroupBox {
    border: 1px solid #3a3f47;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #8ecbff;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #2a2d33;
    color: #e8eaed;
    border: 1px solid #3f4550;
    border-radius: 6px;
    padding: 6px;
}
QComboBox QAbstractItemView {
    background-color: #2a2d33;
    color: #e8eaed;
    selection-background-color: #3f546b;
}
QPushButton {
    background-color: #343942;
    color: #e8eaed;
    border: 1px solid #4a5160;
    border-radius: 8px;
    padding: 6px 10px;
}
QPushButton:hover {
    background-color: #3d4450;
}
QPushButton#themeButton {
    min-width: 30px;
    max-width: 30px;
    padding: 4px;
    border-radius: 15px;
    font-size: 14px;
    color: #8ecbff;
}
"""


class CommandLauncherPyQt(QMainWindow):
    """Alternative launcher using PyQt for UI comparison."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EMaGerLib Command Launcher (PyQt)")
        self.resize(1020, 670)

        self.current_theme = "dark"
        self.config_inputs = {}
        self.config_field_names = [
            f.name for f in fields(CoreConfig)
            if f.name not in {"EXTRA", "CONFIG_FILE_NAME", "CONFIG_FILE_PATH"}
        ]
        self.last_saved_config_path = None

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.readyReadStandardError.connect(self._read_process_output)
        self.process.finished.connect(self._on_process_finished)

        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.addStretch(1)
        self.theme_button = QPushButton("☀")
        self.theme_button.setObjectName("themeButton")
        self.theme_button.clicked.connect(self._toggle_theme)
        top_bar.addWidget(self.theme_button)
        root_layout.addLayout(top_bar)

        emager_group = QGroupBox("Run EMaGer Command")
        emager_layout = QGridLayout(emager_group)

        emager_layout.addWidget(QLabel("Command:"), 0, 0)
        self.command_combo = QComboBox()
        self.command_combo.addItems(get_available_emager_commands())
        emager_layout.addWidget(self.command_combo, 0, 1)

        self.run_selected_button = QPushButton("Run Selected")
        self.run_selected_button.clicked.connect(self._run_selected_command)
        emager_layout.addWidget(self.run_selected_button, 0, 2, 2, 1)

        emager_layout.addWidget(QLabel("Extra args:"), 1, 0)
        self.args_input = QLineEdit()
        self.args_input.setPlaceholderText("--config config_examples/base_config_example.py")
        emager_layout.addWidget(self.args_input, 1, 1)

        root_layout.addWidget(emager_group)

        config_group = QGroupBox("Config Editor")
        config_root_layout = QVBoxLayout(config_group)

        preload_layout = QHBoxLayout()
        preload_layout.addWidget(QLabel("Preload config file:"))
        self.config_path_input = QLineEdit()
        self.config_path_input.setPlaceholderText("config_examples/base_config_example.py")
        self.browse_config_button = QPushButton("Browse")
        self.browse_config_button.clicked.connect(self._browse_config_file)
        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self._load_selected_config_file)
        preload_layout.addWidget(self.config_path_input)
        preload_layout.addWidget(self.browse_config_button)
        preload_layout.addWidget(self.load_config_button)
        config_root_layout.addLayout(preload_layout)

        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Runtime save name:"))
        self.config_name_input = QLineEdit()
        self.config_name_input.setPlaceholderText("gui_runtime_config")
        self.config_name_input.setText("gui_runtime_config")
        save_layout.addWidget(QLabel("Runtime save format:"))
        self.config_format_combo = QComboBox()
        self.config_format_combo.addItems(["yaml", "json"])
        self.config_format_combo.setCurrentText("yaml")
        self.last_config_label = QLabel("Runtime config: not saved yet")
        save_layout.addWidget(self.config_name_input)
        save_layout.addWidget(self.config_format_combo)
        save_layout.addStretch(1)
        save_layout.addWidget(self.last_config_label)
        config_root_layout.addLayout(save_layout)

        self.config_scroll = QScrollArea()
        self.config_scroll.setWidgetResizable(True)
        self.config_scroll_content = QWidget()
        self.config_grid = QGridLayout(self.config_scroll_content)
        self.config_grid.setColumnStretch(1, 1)
        self.config_scroll.setWidget(self.config_scroll_content)
        config_root_layout.addWidget(self.config_scroll)

        root_layout.addWidget(config_group)

        custom_group = QGroupBox("Run Custom Shell Command")
        custom_layout = QHBoxLayout(custom_group)
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("Example: python -m tests.run_all_tests")
        self.run_custom_button = QPushButton("Run Custom")
        self.run_custom_button.clicked.connect(self._run_custom_command)
        custom_layout.addWidget(self.custom_input)
        custom_layout.addWidget(self.run_custom_button)
        root_layout.addWidget(custom_group)

        controls_layout = QHBoxLayout()
        self.stop_button = QPushButton("Stop Running")
        self.stop_button.clicked.connect(self._stop_process)
        self.clear_button = QPushButton("Clear Output")
        self.clear_button.clicked.connect(self._clear_output)
        self.toggle_output_button = QPushButton("Hide Output")
        self.toggle_output_button.clicked.connect(self._toggle_output_visibility)
        self.status_label = QLabel("Idle")

        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.toggle_output_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.status_label)
        root_layout.addLayout(controls_layout)

        self.output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(self.output_group)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        self.output.setMinimumHeight(240)
        output_layout.addWidget(self.output)
        root_layout.addWidget(self.output_group, stretch=2)

        self.setCentralWidget(central)
        self._build_config_editor_fields()
        self._load_initial_config()
        self._apply_theme(self.current_theme)

    def _build_config_editor_fields(self):
        for row, field_name in enumerate(self.config_field_names):
            label = QLabel(field_name)
            input_box = QLineEdit()
            input_box.setPlaceholderText("None")
            self.config_grid.addWidget(label, row, 0)
            self.config_grid.addWidget(input_box, row, 1)
            self.config_inputs[field_name] = input_box

    def _load_initial_config(self):
        default_path = Path("config_examples") / "base_config_example.py"
        self.config_path_input.setText(str(default_path))
        if default_path.exists():
            self._load_config_into_editor(default_path)

    def _browse_config_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select config file",
            str(Path.cwd()),
            "Config Files (*.py *.yaml *.yml *.json)",
        )
        if file_path:
            self.config_path_input.setText(file_path)

    def _load_selected_config_file(self):
        raw_path = self.config_path_input.text().strip()
        if not raw_path:
            self._append_output("Please provide a config file path to preload.\n")
            return
        self._load_config_into_editor(Path(raw_path))

    def _load_config_into_editor(self, config_path: Path):
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            self._append_output(f"Failed to load config '{config_path}': {exc}\n")
            return

        for field_name in self.config_field_names:
            value = getattr(cfg, field_name, None)
            self.config_inputs[field_name].setText(self._value_to_text(value))

        self._append_output(f"Loaded config values from: {Path(config_path).resolve()}\n")

    def _value_to_text(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, str):
            return value
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    def _text_to_value(self, raw_text: str):
        text = raw_text.strip()
        if text == "":
            return ""

        lowered = text.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    def _collect_config_data(self):
        config_data = {}
        for field_name in self.config_field_names:
            raw_value = self.config_inputs[field_name].text()
            value = self._text_to_value(raw_value)
            if field_name == "BASE_PATH" and value is None:
                raise ValueError("BASE_PATH cannot be empty.")
            config_data[field_name] = value
        return config_data

    def _save_runtime_config(self) -> Path:
        config_data = self._collect_config_data()
        saved_dir = Path.cwd() / "saved_configs"
        saved_dir.mkdir(parents=True, exist_ok=True)

        file_format = self.config_format_combo.currentText().strip().lower()
        if file_format not in {"yaml", "json"}:
            file_format = "yaml"

        config_name = self.config_name_input.text().strip() or "gui_runtime_config"

        cfg = CoreConfig(**config_data)
        output_path = save_config(
            cfg,
            saved_dir,
            name=config_name,
            file_format=file_format,
        )

        self.last_saved_config_path = output_path
        self.last_config_label.setText(f"Runtime config: {output_path.name}")
        return output_path

    def _remove_config_args(self, extra_args: str) -> str:
        if not extra_args.strip():
            return ""

        tokens = shlex.split(extra_args, posix=(os.name != "nt"))
        sanitized_tokens = []
        skip_next = False

        for token in tokens:
            if skip_next:
                skip_next = False
                continue

            if token in {"-c", "--config"}:
                skip_next = True
                continue

            if token.startswith("--config="):
                continue

            sanitized_tokens.append(token)

        return " ".join(sanitized_tokens)

    def _apply_theme(self, theme_name):
        self.current_theme = theme_name
        if theme_name == "light":
            self.setStyleSheet(LIGHT_STYLESHEET)
        else:
            self.setStyleSheet(DARK_STYLESHEET)
        self._normalize_widget_fonts()

    def _normalize_widget_fonts(self):
        widgets = [self] + self.findChildren(QWidget)
        for widget in widgets:
            font = widget.font()
            if font.pointSize() > 0 or font.pointSizeF() > 0:
                continue

            pixel_size = font.pixelSize()
            fallback_pt = max(1, int(round(pixel_size * 0.75))) if pixel_size > 0 else 10
            font.setPointSize(fallback_pt)
            widget.setFont(font)

    def _toggle_theme(self):
        next_theme = "light" if self.current_theme == "dark" else "dark"
        self._apply_theme(next_theme)

    def _append_output(self, text):
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)
        self.output.insertPlainText(text)
        self.output.moveCursor(self.output.textCursor().MoveOperation.End)

    def _is_running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _run_selected_command(self):
        command_name = self.command_combo.currentText().strip()
        if not command_name:
            self._append_output("Please select a command.\n")
            return

        try:
            saved_config_path = self._save_runtime_config()
        except Exception as exc:
            self._append_output(f"Failed to save runtime config: {exc}\n")
            return

        sanitized_args = self._remove_config_args(self.args_input.text())
        command = build_emager_command(command_name, sanitized_args)
        command.extend(["--config", str(saved_config_path)])
        self._append_output(f"Using runtime config: {saved_config_path}\n")
        self._start_process(command, shell=False)

    def _run_custom_command(self):
        custom_command = self.custom_input.text().strip()
        if not custom_command:
            self._append_output("Please provide a custom command.\n")
            return
        self._start_process(custom_command, shell=True)

    def _start_process(self, command_line, shell=False):
        if self._is_running():
            self._append_output("A process is already running. Stop it before starting another one.\n")
            return

        self._append_output("\n$ {}\n".format(command_line))
        self.status_label.setText("Running")

        if shell:
            if os.name == "nt":
                self.process.start("cmd", ["/c", command_line])
            else:
                self.process.start("/bin/sh", ["-c", command_line])
        else:
            program = command_line[0]
            arguments = command_line[1:]
            self.process.start(program, arguments)

        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.status_label.setText("Error")
            self._append_output("Failed to start command.\n")

    def _read_process_output(self):
        stdout_data = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        stderr_data = bytes(self.process.readAllStandardError()).decode(errors="replace")
        if stdout_data:
            self._append_output(stdout_data)
        if stderr_data:
            self._append_output(stderr_data)

    def _on_process_finished(self, exit_code, _exit_status):
        self._append_output("\nProcess finished with exit code {}.\n".format(exit_code))
        self.status_label.setText("Idle")

    def _stop_process(self):
        if not self._is_running():
            self._append_output("No running process to stop.\n")
            return
        self.process.kill()
        self._append_output("Stop signal sent.\n")
        self.status_label.setText("Stopping")

    def _clear_output(self):
        self.output.clear()

    def _toggle_output_visibility(self):
        is_visible = self.output_group.isVisible()
        self.output_group.setVisible(not is_visible)
        if is_visible:
            self.toggle_output_button.setText("Show Output")
        else:
            self.toggle_output_button.setText("Hide Output")


def main():
    """Entry point for the PyQt launcher."""
    if QApplication is None:
        raise RuntimeError("PyQt6 is required for this launcher. Install pyqt6 to run it.")

    app = QApplication.instance() or QApplication(sys.argv)
    window = CommandLauncherPyQt()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
