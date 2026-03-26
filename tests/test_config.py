"""Test suite for config system"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from emagerlib.config.load_config import load_config
from emagerlib.config.save_config import save_config
from emagerlib import ROOT_EMAGERLIB


class TestConfigSystem(unittest.TestCase):
    """Test suite for configuration system"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.config_dir = Path(__file__).parent.parent / "config_examples"
        cls.test_dir = Path(__file__).parent
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)
    
    def test_01_load_python_config(self):
        """Test loading Python config file"""
        config_file = self.config_dir / "base_config_example.py"
        
        # Just verify loading works and returns a config object
        cfg = load_config(config_file)
        
        self.assertIsNotNone(cfg, "Config should not be None")
        self.assertTrue(hasattr(cfg, 'SESSION'), "Config should have SESSION attribute")
        self.assertTrue(hasattr(cfg, 'SAMPLING'), "Config should have SAMPLING attribute")
        print(f"✓ Python config loaded successfully from {config_file.name}")

    def test_02_save_and_load_json(self):
        """Test round-trip: Python -> JSON -> Load"""
        # Load from Python
        config_file = self.config_dir / "base_config_example.py"
        cfg_original = load_config(config_file)
        
        # Get some original values to compare
        original_data = {
            'SESSION': cfg_original.SESSION,
            'SAMPLING': cfg_original.SAMPLING,
            'CLASSES': cfg_original.CLASSES,
            'SMOOTH_METHOD': cfg_original.SMOOTH_METHOD
        }
        
        # Save to JSON (default format)
        saved_file = save_config(cfg_original, self.test_dir, name="test_config")
        self.assertTrue(saved_file.exists(), "Saved file should exist")
        self.assertTrue(saved_file.suffix == ".json", "Default format should be JSON")
        print(f"✓ Config saved to {saved_file.name}")
        
        try:
            # Load from JSON
            cfg_loaded = load_config(saved_file)
            self.assertIsNotNone(cfg_loaded, "Loaded config should not be None")
            print(f"✓ Config loaded from {saved_file.name}")
            
            # Verify round-trip worked (data is preserved)
            for key, original_value in original_data.items():
                loaded_value = getattr(cfg_loaded, key)
                self.assertEqual(loaded_value, original_value, 
                               f"Round-trip failed: {key} changed from {original_value} to {loaded_value}")
            
            print("✓ Round-trip successful: all data preserved")
            
        finally:
            # Cleanup
            if saved_file.exists():
                saved_file.unlink()

    def test_03_load_yaml(self):
        """Test loading YAML config file"""
        config_file = self.config_dir / "base_config_example.yaml"
        
        # Just verify loading works
        cfg = load_config(config_file)
        
        self.assertIsNotNone(cfg, "Config should not be None")
        self.assertTrue(hasattr(cfg, 'SESSION'), "Config should have SESSION attribute")
        self.assertTrue(hasattr(cfg, 'SAMPLING'), "Config should have SAMPLING attribute")
        print(f"✓ YAML config loaded successfully from {config_file.name}")

    def test_04_extra_fields(self):
        """Test that custom fields can be added and accessed"""
        config_file = self.config_dir / "base_config_example.py"
        cfg = load_config(config_file)
        
        # Add custom fields
        cfg.CUSTOM_FIELD = "test_value"
        cfg.CUSTOM_NUMBER = 42
        
        # Verify they're accessible
        self.assertEqual(cfg.CUSTOM_FIELD, "test_value")
        self.assertEqual(cfg.CUSTOM_NUMBER, 42)
        self.assertEqual(cfg.get("CUSTOM_FIELD"), "test_value")
        print("✓ Custom fields can be added and accessed")

    def test_05_save_and_load_yaml(self):
        """Test saving config as YAML and loading it back"""
        config_file = self.config_dir / "base_config_example.py"
        cfg_original = load_config(config_file)
        
        save_dir = Path(self.temp_dir) / "yaml_test"
        
        # Save as YAML
        saved_file = save_config(
            cfg_original,
            save_dir,
            name="test_config",
            file_format="yaml"
        )
        
        self.assertTrue(saved_file.exists())
        self.assertTrue(saved_file.suffix == ".yaml")
        
        # Load it back
        cfg_loaded = load_config(saved_file)
        
        # Verify data matches
        self.assertEqual(cfg_loaded.SESSION, cfg_original.SESSION)
        self.assertEqual(cfg_loaded.SAMPLING, cfg_original.SAMPLING)
        self.assertEqual(cfg_loaded.CLASSES, cfg_original.CLASSES)
        print("✓ YAML save/load round-trip successful")

    def test_06_save_creates_directory(self):
        """Test that save creates directories if they don't exist"""
        save_dir = Path(self.temp_dir) / "deep" / "nested" / "directory"
        
        self.assertFalse(save_dir.exists())
        
        config_file = self.config_dir / "base_config_example.py"
        cfg = load_config(config_file)
        
        saved_file = save_config(cfg, save_dir, name="nested_config")
        
        self.assertTrue(save_dir.exists())
        self.assertTrue(saved_file.exists())
        print("✓ Save creates nested directories")

    def test_07_save_with_timestamp(self):
        """Test that saved files include timestamp"""
        save_dir = Path(self.temp_dir) / "timestamp_test"
        
        config_file = self.config_dir / "base_config_example.py"
        cfg = load_config(config_file)
        
        saved_file = save_config(cfg, save_dir, name="timestamped")
        
        # Filename should contain timestamp pattern (YYYYMMDD_HHMMSS)
        filename = saved_file.stem  # Without extension
        self.assertTrue("_" in filename)
        
        # Should have at least two parts: name and timestamp
        parts = filename.split("_")
        self.assertGreaterEqual(len(parts), 2)
        print("✓ Saved files include timestamp")

    def test_08_python_config_independent_of_cwd(self):
        """Test Python config path resolution is stable across different CWDs."""
        config_file = self.config_dir / "base_config_example.py"
        expected_base = (ROOT_EMAGERLIB / "Datasets").resolve()
        expected_media = (ROOT_EMAGERLIB / "media-test").resolve()

        original_cwd = Path.cwd()
        candidate_cwds = [
            self.test_dir,
            self.config_dir,
            ROOT_EMAGERLIB,
        ]

        try:
            for cwd in candidate_cwds:
                os.chdir(cwd)
                cfg = load_config(config_file)
                self.assertEqual(cfg.BASE_PATH.resolve(), expected_base)
                self.assertEqual(Path(cfg.MEDIA_PATH).resolve(), expected_media)
        finally:
            os.chdir(original_cwd)

        print("✓ Python config paths are independent of CWD")

    def test_09_yaml_config_independent_of_cwd(self):
        """Test YAML config path resolution is stable across different CWDs."""
        config_file = self.config_dir / "base_config_example.yaml"
        expected_base = (ROOT_EMAGERLIB / "Datasets").resolve()
        expected_media = (ROOT_EMAGERLIB / "media-test").resolve()

        original_cwd = Path.cwd()
        candidate_cwds = [
            self.test_dir,
            self.config_dir,
            ROOT_EMAGERLIB,
        ]

        try:
            for cwd in candidate_cwds:
                os.chdir(cwd)
                cfg = load_config(config_file)
                self.assertEqual(cfg.BASE_PATH.resolve(), expected_base)
                self.assertEqual(Path(cfg.MEDIA_PATH).resolve(), expected_media)
        finally:
            os.chdir(original_cwd)

        print("✓ YAML config paths are independent of CWD")

    def test_10_path_fields_are_normalized_to_path_objects(self):
        """Path-like config fields should be loaded as pathlib.Path objects."""
        cfg_py = load_config(self.config_dir / "base_config_example.py")
        cfg_yaml = load_config(self.config_dir / "base_config_example.yaml")

        for cfg in (cfg_py, cfg_yaml):
            self.assertIsInstance(cfg.BASE_PATH, Path)
            self.assertIsInstance(cfg.MEDIA_PATH, Path)
            self.assertIsInstance(cfg.PRETRAINED_MODEL_PATH, Path)

        print("✓ Path-like fields are normalized to Path objects")

    def test_11_media_path_join_does_not_break_separator(self):
        """Regression test for broken joins like 'media-testgesture_list.json'."""
        cfg_py = load_config(self.config_dir / "base_config_example.py")
        cfg_yaml = load_config(self.config_dir / "base_config_example.yaml")

        for cfg in (cfg_py, cfg_yaml):
            gestures_file = cfg.MEDIA_PATH / "gesture_list.json"
            self.assertEqual(gestures_file.parent, cfg.MEDIA_PATH)
            self.assertEqual(gestures_file.name, "gesture_list.json")
            self.assertNotIn("media-testgesture_list.json", str(gestures_file))

        print("✓ MEDIA_PATH joins keep proper separators")

if __name__ == '__main__':
    unittest.main()