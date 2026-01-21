"""Test suite for config system"""

import unittest
from pathlib import Path
from emager_tools.config.loader import load_py_config, load_json_config, load_yaml_config
from emager_tools.config.util import save_config


class TestConfigSystem(unittest.TestCase):
    """Test suite for configuration system"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.config_dir = Path(__file__).parent.parent / "config_examples"
        cls.test_dir = Path(__file__).parent
    
    def test_01_load_python_config(self):
        """Test loading Python config file"""
        config_file = self.config_dir / "base_config_example.py"
        
        # Just verify loading works and returns a config object
        cfg = load_py_config(config_file)
        
        self.assertIsNotNone(cfg, "Config should not be None")
        self.assertTrue(hasattr(cfg, 'SESSION'), "Config should have SESSION attribute")
        self.assertTrue(hasattr(cfg, 'SAMPLING'), "Config should have SAMPLING attribute")
        print(f"✓ Python config loaded successfully from {config_file.name}")

    def test_02_save_and_load_json(self):
        """Test round-trip: Python -> JSON -> Load"""
        # Load from Python
        config_file = self.config_dir / "base_config_example.py"
        cfg_original = load_py_config(config_file)
        
        # Get some original values to compare
        original_data = {
            'SESSION': cfg_original.SESSION,
            'SAMPLING': cfg_original.SAMPLING,
            'CLASSES': cfg_original.CLASSES,
            'SMOOTH_METHOD': cfg_original.SMOOTH_METHOD
        }
        
        # Save to JSON
        saved_file = save_config(cfg_original, self.test_dir, name="test_config")
        self.assertTrue(saved_file.exists(), "Saved file should exist")
        print(f"✓ Config saved to {saved_file.name}")
        
        try:
            # Load from JSON
            cfg_loaded = load_json_config(saved_file)
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
        cfg = load_yaml_config(config_file)
        
        self.assertIsNotNone(cfg, "Config should not be None")
        self.assertTrue(hasattr(cfg, 'SESSION'), "Config should have SESSION attribute")
        self.assertTrue(hasattr(cfg, 'SAMPLING'), "Config should have SAMPLING attribute")
        print(f"✓ YAML config loaded successfully from {config_file.name}")

    def test_04_extra_fields(self):
        """Test that custom fields can be added and accessed"""
        config_file = self.config_dir / "base_config_example.py"
        cfg = load_py_config(config_file)
        
        # Add custom fields
        cfg.CUSTOM_FIELD = "test_value"
        cfg.CUSTOM_NUMBER = 42
        
        # Verify they're accessible
        self.assertEqual(cfg.CUSTOM_FIELD, "test_value")
        self.assertEqual(cfg.CUSTOM_NUMBER, 42)
        self.assertEqual(cfg.get("CUSTOM_FIELD"), "test_value")
        print("✓ Custom fields can be added and accessed")

if __name__ == '__main__':
    unittest.main()