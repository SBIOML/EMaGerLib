"""Test suite for gesture JSON utilities"""

import unittest
from pathlib import Path
import tempfile
import shutil
import json
from emager_tools.utils.gestures_json import (
    get_images_list,
    get_images_folder,
    get_gestures_dict,
    get_index_from_label,
    get_label_from_index
)


class TestGesturesJson(unittest.TestCase):
    """Test suite for gesture JSON utilities"""

    @classmethod
    def setUpClass(cls):
        """Create temporary directory structure with test images and JSON"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.images_folder = Path(cls.temp_dir) / "gestures"
        cls.images_folder.mkdir()
        
        # Create test image files
        (cls.images_folder / "gesture_1.png").touch()
        (cls.images_folder / "gesture_2.png").touch()
        (cls.images_folder / "gesture_3.jpg").touch()
        (cls.images_folder / "readme.txt").touch()  # Non-image file
        
        # Create test gestures JSON
        cls.gestures_dict = {
            "1": "gesture_1",
            "2": "gesture_2",
            "3": "gesture_3"
        }
        
        with open(cls.images_folder / "gesture_list.json", "w") as f:
            json.dump(cls.gestures_dict, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_01_get_images_list(self):
        """Test getting list of image files from folder"""
        images = get_images_list(str(self.images_folder) + "/")
        
        # Should find 3 images (.png and .jpg only, not .txt)
        self.assertEqual(len(images), 3)
        self.assertTrue(all(img.endswith(('.png', '.jpg')) for img in images))
        print("✓ get_images_list returns only image files")

    def test_02_get_images_folder(self):
        """Test extracting folder from image list"""
        images = get_images_list(str(self.images_folder) + "/")
        folder = get_images_folder(images)
        
        self.assertEqual(Path(folder), self.images_folder)
        print("✓ get_images_folder extracts correct folder")

    def test_03_get_images_folder_empty_list(self):
        """Test get_images_folder with empty list raises error"""
        with self.assertRaises(ValueError):
            get_images_folder([])
        print("✓ get_images_folder raises ValueError for empty list")

    def test_04_get_images_folder_nonexistent_file(self):
        """Test get_images_folder with non-existent file raises error"""
        with self.assertRaises(FileNotFoundError):
            get_images_folder(["/nonexistent/path/image.png"])
        print("✓ get_images_folder raises FileNotFoundError for missing files")

    def test_05_get_gestures_dict_from_folder(self):
        """Test loading gestures dictionary from folder"""
        gestures = get_gestures_dict(str(self.images_folder))
        
        self.assertIsNotNone(gestures)
        self.assertEqual(gestures["1"], "gesture_1")
        self.assertEqual(gestures["2"], "gesture_2")
        print("✓ get_gestures_dict loads from folder correctly")

    def test_06_get_gestures_dict_from_image_list(self):
        """Test loading gestures dictionary from image list"""
        images = get_images_list(str(self.images_folder) + "/")
        gestures = get_gestures_dict(images)
        
        self.assertIsNotNone(gestures)
        self.assertEqual(len(gestures), 3)
        print("✓ get_gestures_dict loads from image list correctly")

    def test_07_get_gestures_dict_nonexistent_folder(self):
        """Test get_gestures_dict with non-existent folder raises error"""
        with self.assertRaises(FileNotFoundError):
            get_gestures_dict("/nonexistent/folder")
        print("✓ get_gestures_dict raises FileNotFoundError for missing folder")

    def test_08_get_index_from_label(self):
        """Test finding image index from gesture label"""
        images = get_images_list(str(self.images_folder) + "/")
        index = get_index_from_label(1, images, self.gestures_dict)
        
        self.assertIsNotNone(index)
        self.assertTrue("gesture_1" in images[index])
        print("✓ get_index_from_label finds correct index")

    def test_09_get_label_from_index(self):
        """Test finding gesture label from image index"""
        images = get_images_list(str(self.images_folder) + "/")
        
        # Find index of gesture_2
        index = None
        for i, img in enumerate(images):
            if "gesture_2" in img:
                index = i
                break
        
        if index is not None:
            label = get_label_from_index(index, images, self.gestures_dict)
            self.assertEqual(label, 2)
            print("✓ get_label_from_index finds correct label")
        else:
            self.fail("Could not find gesture_2 in images list")


if __name__ == '__main__':
    unittest.main()
