"""Test suite for model finding utilities"""

import unittest
from pathlib import Path
import tempfile
import shutil
from emager_tools.utils.find_models import find_models, find_last_model


class TestFindModels(unittest.TestCase):
    """Test suite for finding model files"""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory structure for testing"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.base_path = Path(cls.temp_dir)
        cls.session_dir = cls.base_path / "test_session"
        cls.session_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_01_find_models_empty_directory(self):
        """Test finding models in empty directory"""
        models = find_models(str(self.base_path), "test_session")
        
        self.assertEqual(models, [])
        print("✓ find_models returns empty list for empty directory")

    def test_02_find_models_with_files(self):
        """Test finding models with .pth files"""
        # Create some test model files
        (self.session_dir / "model_01-01-25_10h30.pth").touch()
        (self.session_dir / "model_02-01-25_14h45.pth").touch()
        (self.session_dir / "not_a_model.txt").touch()
        
        models = find_models(str(self.base_path), "test_session")
        
        self.assertEqual(len(models), 2)
        self.assertTrue(all(m.endswith('.pth') for m in models))
        print("✓ find_models returns only .pth files")

    def test_03_find_models_nonexistent_session(self):
        """Test finding models in non-existent session"""
        models = find_models(str(self.base_path), "nonexistent_session")
        
        self.assertEqual(models, [])
        print("✓ find_models handles non-existent sessions")

    def test_04_find_last_model_no_models(self):
        """Test finding last model when no models exist"""
        empty_session = self.base_path / "empty_session"
        empty_session.mkdir()
        
        last_model = find_last_model(str(self.base_path), "empty_session")
        
        self.assertIsNone(last_model)
        print("✓ find_last_model returns None when no models exist")

    def test_05_find_last_model_ordering(self):
        """Test that find_last_model returns the most recent model"""
        # Create models with different timestamps (format: dd-mm-yy_HHhMM)
        test_session = self.base_path / "ordered_session"
        test_session.mkdir()
        
        (test_session / "model_01-01-25_10h30.pth").touch()  # Jan 1, 10:30
        (test_session / "model_02-01-25_14h45.pth").touch()  # Jan 2, 14:45
        (test_session / "model_03-01-25_09h15.pth").touch()  # Jan 3, 09:15 (newest)
        
        last_model = find_last_model(str(self.base_path), "ordered_session")
        
        self.assertIsNotNone(last_model)
        # Should return the newest one (03-01-25 = January 3rd)
        self.assertIn("03-01-25", last_model)
        print("✓ find_last_model returns most recent model")


if __name__ == '__main__':
    unittest.main()
