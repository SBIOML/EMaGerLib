"""Test suite for argument parser and logging utilities"""

import unittest
import tempfile
import shutil
import logging
import time
from pathlib import Path
from argparse import Namespace

from emagerlib.utils.arg_parser import create_parser, setup_logging, save_config_if_requested
from emagerlib.config.load_config import load_config


class TestArgumentParser(unittest.TestCase):
    """Test suite for argument parser creation"""

    def test_01_create_base_parser_default(self):
        """Test creating parser with default config"""
        parser = create_parser("Test description")
        
        self.assertIsNotNone(parser)
        # Parse with no arguments (should use defaults)
        args = parser.parse_args([])
        
        self.assertTrue(hasattr(args, 'config'))
        self.assertTrue(hasattr(args, 'save_config_path'))
        self.assertTrue(hasattr(args, 'save_config_name'))
        self.assertTrue(hasattr(args, 'save_config_format'))
        self.assertTrue(hasattr(args, 'log_file_path'))
        self.assertTrue(hasattr(args, 'log_level'))
        self.assertTrue(hasattr(args, 'log_file_name'))
        self.assertTrue(hasattr(args, 'log_to_file'))
        self.assertTrue(hasattr(args, 'no_log_to_file'))
        print("[PASS] Base parser created with all expected arguments")

    def test_02_create_base_parser_custom_config(self):
        """Test creating parser with custom default config"""
        custom_config = "path/to/custom_config.py"
        parser = create_parser("Test", default_config=custom_config)
        args = parser.parse_args([])
        
        self.assertEqual(args.config, custom_config)
        print("[PASS] Parser accepts custom default config")

    def test_03_parse_config_argument(self):
        """Test parsing config file argument"""
        parser = create_parser("Test")
        args = parser.parse_args(["-c", "my_config.yaml"])
        
        self.assertEqual(args.config, "my_config.yaml")
        print("[PASS] Config argument parsed correctly")

    def test_04_parse_save_config_arguments(self):
        """Test parsing config save arguments"""
        parser = create_parser("Test")
        args = parser.parse_args([
            "--save-config-path", "output.json",
            "--save-config-name", "test_config",
            "--save-config-format", "yaml"
        ])
        
        self.assertEqual(args.save_config_path, "output.json")
        self.assertEqual(args.save_config_name, "test_config")
        self.assertEqual(args.save_config_format, "yaml")
        print("[PASS] Config save arguments parsed correctly")

    def test_05_parse_logging_arguments(self):
        """Test parsing logging arguments"""
        parser = create_parser("Test")
        args = parser.parse_args([
            "--log-file-path", "test.log",
            "--log-level", "DEBUG",
            "--log-file-name", "my_log"
        ])
        
        self.assertEqual(args.log_file_path, "test.log")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.log_file_name, "my_log")
        print("[PASS] Logging arguments parsed correctly")


class TestLoggingSetup(unittest.TestCase):
    """Test suite for logging setup functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up temporary directory for log files"""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)
        # Reset logging to default state
        logging.getLogger().handlers.clear()

    def test_01_setup_logging_console_only(self):
        """Test logging setup with console output only"""
        args = Namespace(
            log_file_path=None,
            log_file_name=None,
            log_level="INFO",
            log_to_file=None,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=None, script_name="test")
        
        # Should have at least one handler (console)
        self.assertGreaterEqual(len(logging.getLogger().handlers), 1)
        self.assertEqual(logging.getLogger().level, logging.INFO)
        print("[PASS] Console-only logging setup works")

    def test_02_setup_logging_with_file(self):
        """Test logging setup with file output"""
        log_file = Path(self.temp_dir) / "test.log"
        args = Namespace(
            log_file_path=str(log_file),
            log_file_name=None,
            log_level="DEBUG",
            log_to_file=None,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=None, script_name="test")
        
        # Should have at least two handlers (console + file)
        self.assertGreaterEqual(len(logging.getLogger().handlers), 2)
        self.assertTrue(log_file.exists())
        print("[PASS] File logging setup works")

    def test_03_setup_logging_with_log_name(self):
        """Test logging setup with auto-generated log name"""
        args = Namespace(
            log_file_path=None,
            log_file_name="custom_log.log",
            log_level="WARNING",
            log_to_file=True,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=None, script_name="test")
        
        # Check that log file was created in logs directory
        log_dir = Path.cwd() / "logs"
        self.assertTrue(log_dir.exists())
        
        # Find the created log file
        log_files = list(log_dir.glob("custom_log.log"))
        self.assertGreater(len(log_files), 0)
        
        # Close all handlers to release file locks
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # Clean up
        for log_file in log_files:
            try:
                log_file.unlink()
            except PermissionError:
                pass  # Skip if still locked
        
        print("[PASS] Auto-generated log name works")


class TestConfigSaving(unittest.TestCase):
    """Test suite for configuration saving functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up temporary directory and load a config"""
        cls.temp_dir = tempfile.mkdtemp()
        config_dir = Path(__file__).parent.parent / "config_examples"
        cls.config = load_config(config_dir / "base_config_example.py")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_01_save_config_not_requested(self):
        """Test that config is not saved when not requested"""
        args = Namespace(
            save_config_path=None,
            save_config_name=None,
            save_config_format=None
        )
        
        # Should not raise any errors and not save anything
        save_config_if_requested(args, self.config)
        print("[PASS] Config not saved when not requested")

    def test_02_save_config_with_path(self):
        """Test saving config to specific path"""
        save_path = Path(self.temp_dir) / "test_config.json"
        args = Namespace(
            save_config_path=str(save_path),
            save_config_name=None,
            save_config_format="json"
        )
        
        save_config_if_requested(args, self.config)
        
        # Check that multiple files exist (because of timestamp)
        saved_files = list(Path(self.temp_dir).glob("test_config_*.json"))
        self.assertGreater(len(saved_files), 0)
        print("[PASS] Config saved to specific path")

    def test_03_save_config_with_name(self):
        """Test saving config with custom name"""
        args = Namespace(
            save_config_path=None,
            save_config_name="my_experiment",
            save_config_format="json"
        )
        
        save_config_if_requested(args, self.config)
        
        # Check in default saved_configs directory
        save_dir = Path.cwd() / "saved_configs"
        self.assertTrue(save_dir.exists())
        
        saved_files = list(save_dir.glob("my_experiment_*.json"))
        self.assertGreater(len(saved_files), 0)
        
        # Clean up
        for f in saved_files:
            f.unlink()
        
        print("[PASS] Config saved with custom name")

    def test_04_save_config_yaml_format(self):
        """Test saving config in YAML format"""
        # Clean up any existing yaml_test files first
        save_dir = Path.cwd() / "saved_configs"
        if save_dir.exists():
            for old_file in save_dir.glob("yaml_test_*.yaml"):
                old_file.unlink()
        
        args = Namespace(
            save_config_path=None,
            save_config_name="yaml_test",
            save_config_format="yaml"
        )
        
        save_config_if_requested(args, self.config)
        
        saved_files = list(save_dir.glob("yaml_test_*.yaml"))
        self.assertGreater(len(saved_files), 0)
        
        # Verify it's valid YAML and can be loaded back
        from emagerlib.config.load_config import load_config
        loaded = load_config(saved_files[0])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.SESSION, self.config.SESSION)
        
        # Clean up
        for f in saved_files:
            f.unlink()
        
        print("[PASS] Config saved in YAML format")


class TestFallbackMechanism(unittest.TestCase):
    """Test suite for config fallback mechanism in arg parser"""

    @classmethod
    def setUpClass(cls):
        """Set up temporary directory and load a config"""
        cls.temp_dir = tempfile.mkdtemp()
        config_dir = Path(__file__).parent.parent / "config_examples"
        cls.config = load_config(config_dir / "base_config_example.py")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)
        # Reset logging to default state
        logging.getLogger().handlers.clear()

    def test_01_logging_fallback_to_config(self):
        """Test that logging uses config defaults when args are None"""
        # Create a config with specific logging settings
        self.config.LOG_LEVEL = "DEBUG"
        self.config.LOG_TO_FILE = True
        self.config.LOG_FILE_NAME = "test_fallback.log"
        
        # Args with all None values
        args = Namespace(
            log_file_path=None,
            log_file_name=None,
            log_level=None,
            log_to_file=None,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=self.config, script_name="test_fallback")
        
        # Should use config's DEBUG level
        self.assertEqual(logging.getLogger().level, logging.DEBUG)
        
        # Should have file handler because config.LOG_TO_FILE = True
        self.assertGreaterEqual(len(logging.getLogger().handlers), 2)
        
        # Close all handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # Clean up log file
        log_dir = Path.cwd() / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("test_fallback.log"):
                try:
                    log_file.unlink()
                except PermissionError:
                    pass
        
        print("[PASS] Logging falls back to config defaults")

    def test_02_logging_args_override_config(self):
        """Test that args override config when provided"""
        # Config has DEBUG, but args specify INFO
        self.config.LOG_LEVEL = "DEBUG"
        
        args = Namespace(
            log_file_path=None,
            log_file_name=None,
            log_level="WARNING",
            log_to_file=None,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=self.config, script_name="test_override")
        
        # Should use args' WARNING level, not config's DEBUG
        self.assertEqual(logging.getLogger().level, logging.WARNING)
        
        # Close all handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        print("[PASS] Args override config for logging")

    def test_03_save_config_fallback_to_config(self):
        """Test that save_config uses config defaults when args are None"""
        # Set config defaults
        self.config.SAVE_CONFIG_NAME = "config_from_defaults"
        self.config.SAVE_CONFIG_FORMAT = "yaml"
        
        # Args with all None values
        args = Namespace(
            save_config_path=None,
            save_config_name=None,
            save_config_format=None
        )
        
        save_config_if_requested(args, self.config)
        
        # Check that config was saved using config defaults
        save_dir = Path.cwd() / "saved_configs"
        saved_files = list(save_dir.glob("config_from_defaults_*.yaml"))
        self.assertGreater(len(saved_files), 0)
        
        # Clean up
        for f in saved_files:
            f.unlink()
        
        print("[PASS] Config saving falls back to config defaults")

    def test_04_save_config_args_override_config(self):
        """Test that args override config for saving"""
        # Config has yaml, but args specify json
        self.config.SAVE_CONFIG_NAME = "config_from_defaults"
        self.config.SAVE_CONFIG_FORMAT = "yaml"
        
        args = Namespace(
            save_config_path=None,
            save_config_name="config_from_args",
            save_config_format="json"
        )
        
        save_config_if_requested(args, self.config)
        
        # Should use args' values, not config's
        save_dir = Path.cwd() / "saved_configs"
        saved_files = list(save_dir.glob("config_from_args_*.json"))
        self.assertGreater(len(saved_files), 0)
        
        # Should NOT find yaml files with config name
        yaml_files = list(save_dir.glob("config_from_defaults_*.yaml"))
        # Filter to only files created in this test
        recent_yaml = [f for f in yaml_files if f.stat().st_mtime > (time.time() - 60)]
        self.assertEqual(len(recent_yaml), 0)
        
        # Clean up
        for f in saved_files:
            f.unlink()
        
        print("[PASS] Args override config for saving")

    def test_05_no_log_to_file_flag_overrides_config(self):
        """Test that --no-log-to-file overrides config.LOG_TO_FILE=True"""
        self.config.LOG_TO_FILE = True
        
        args = Namespace(
            log_file_path=None,
            log_file_name=None,
            log_level=None,
            log_to_file=None,
            no_log_to_file=True
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=self.config, script_name="test_no_file")
        
        # Should only have console handler (1), not file handler
        self.assertEqual(len(logging.getLogger().handlers), 1)
        
        # Close all handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        print("[PASS] --no-log-to-file flag overrides config")

    def test_06_log_to_file_flag_overrides_config(self):
        """Test that --log-to-file overrides config.LOG_TO_FILE=False"""
        self.config.LOG_TO_FILE = False
        
        args = Namespace(
            log_file_path=None,
            log_file_name=None,
            log_level=None,
            log_to_file=True,
            no_log_to_file=None
        )
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        setup_logging(args, cfg=self.config, script_name="test_with_file")
        
        # Should have both console and file handlers
        self.assertGreaterEqual(len(logging.getLogger().handlers), 2)
        
        # Close all handlers
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)
        
        # Clean up
        log_dir = Path.cwd() / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("test_with_file_*.log"):
                try:
                    log_file.unlink()
                except PermissionError:
                    pass
        
        print("[PASS] --log-to-file flag overrides config")


if __name__ == '__main__':
    unittest.main()
