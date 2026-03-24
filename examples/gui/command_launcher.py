"""PyQt-based GUI command launcher for EMaGerLib (comparison implementation)."""

import os
import shlex
import subprocess
import sys

try:
    from PyQt6.QtCore import QProcess
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QApplication,
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
    )
except ImportError:  # pragma: no cover
    QProcess = None
    QApplication = None


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
    font-size: 12px;
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
    font-size: 12px;
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
        output_layout.addWidget(self.output)
        root_layout.addWidget(self.output_group, stretch=1)

        self.setCentralWidget(central)
        self._apply_theme(self.current_theme)

    def _apply_theme(self, theme_name):
        self.current_theme = theme_name
        if theme_name == "light":
            self.setStyleSheet(LIGHT_STYLESHEET)
        else:
            self.setStyleSheet(DARK_STYLESHEET)

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

        command = build_emager_command(command_name, self.args_input.text())
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
