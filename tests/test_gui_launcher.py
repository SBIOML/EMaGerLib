import unittest

from examples.gui.command_launcher import (
    build_emager_command,
    get_available_emager_commands,
    split_arguments,
)


class TestGuiLauncherHelpers(unittest.TestCase):
    def test_split_arguments_empty(self):
        self.assertEqual(split_arguments(""), [])

    def test_split_arguments_with_values(self):
        parsed = split_arguments("--config config_examples/base_config_example.py --log-level DEBUG")
        self.assertEqual(
            parsed,
            ["--config", "config_examples/base_config_example.py", "--log-level", "DEBUG"],
        )

    def test_get_available_commands_contains_known_command(self):
        commands = get_available_emager_commands()
        self.assertIn("train-cnn", commands)
        self.assertNotIn("gui", commands)

    def test_build_emager_command(self):
        command = build_emager_command("realtime-predict", "--log-level INFO")
        self.assertEqual(command[1:4], ["-m", "examples.main", "realtime-predict"])
        self.assertEqual(command[-2:], ["--log-level", "INFO"])


if __name__ == "__main__":
    unittest.main()
