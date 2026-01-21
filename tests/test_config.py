"""Test suite for config system"""

from pathlib import Path
from emager_tools.config.loader import load_py_config, load_json_config, load_yaml_config
from emager_tools.config.util import save_config, print_config


def test_base_example():
    """Test loading config_examples/base_config_example.py"""
    
    config_file = Path(__file__).parent.parent / "config_examples" / "base_config_example.py"
    
    print("="*70)
    print(f"TEST 1: Loading Python config from {config_file.name}")
    print("="*70)
    
    cfg = load_py_config(config_file)
    
    # Test core fields
    print("\n--- Core Fields ---")
    assert cfg.BASE_PATH == Path("./Datasets/"), f"BASE_PATH mismatch: {cfg.BASE_PATH}"
    assert cfg.SESSION == "D1", f"SESSION mismatch: {cfg.SESSION}"
    assert cfg.SAMPLING == 1010, f"SAMPLING mismatch: {cfg.SAMPLING}"
    assert cfg.WINDOW_SIZE == 200, f"WINDOW_SIZE mismatch: {cfg.WINDOW_SIZE}"
    assert cfg.WINDOW_INCREMENT == 10, f"WINDOW_INCREMENT mismatch: {cfg.WINDOW_INCREMENT}"
    assert cfg.FILTER == False, f"FILTER mismatch: {cfg.FILTER}"
    assert cfg.VIRTUAL == False, f"VIRTUAL mismatch: {cfg.VIRTUAL}"
    assert cfg.PORT is None, f"PORT mismatch: {cfg.PORT}"
    print(f"✓ BASE_PATH: {cfg.BASE_PATH}")
    print(f"✓ SESSION: {cfg.SESSION}")
    print(f"✓ SAMPLING: {cfg.SAMPLING}")
    print(f"✓ WINDOW_SIZE: {cfg.WINDOW_SIZE}")
    print(f"✓ FILTER: {cfg.FILTER}")
    
    # Test gesture/class fields
    print("\n--- Gesture/Class Configuration ---")
    assert cfg.NUM_CLASSES == 5, f"NUM_CLASSES mismatch: {cfg.NUM_CLASSES}"
    assert cfg.CLASSES == [2, 3, 30, 14, 18], f"CLASSES mismatch: {cfg.CLASSES}"
    assert cfg.NUM_REPS == 5, f"NUM_REPS mismatch: {cfg.NUM_REPS}"
    assert cfg.REP_TIME == 5, f"REP_TIME mismatch: {cfg.REP_TIME}"
    assert cfg.REST_TIME == 1, f"REST_TIME mismatch: {cfg.REST_TIME}"
    print(f"✓ NUM_CLASSES: {cfg.NUM_CLASSES}")
    print(f"✓ CLASSES: {cfg.CLASSES}")
    print(f"✓ NUM_REPS: {cfg.NUM_REPS}")
    
    # Test model parameters
    print("\n--- Model Parameters ---")
    assert cfg.MAJORITY_VOTE == 30, f"MAJORITY_VOTE mismatch: {cfg.MAJORITY_VOTE}"
    assert cfg.EPOCH == 10, f"EPOCH mismatch: {cfg.EPOCH}"
    print(f"✓ MAJORITY_VOTE: {cfg.MAJORITY_VOTE}")
    print(f"✓ EPOCH: {cfg.EPOCH}")
    
    # Test controller settings
    print("\n--- Controller Settings ---")
    assert cfg.USE_GUI == False, f"USE_GUI mismatch: {cfg.USE_GUI}"
    assert cfg.POLL_SLEEP_DELAY == 0.001, f"POLL_SLEEP_DELAY mismatch: {cfg.POLL_SLEEP_DELAY}"
    assert cfg.PREDICTOR_DELAY == 0.01, f"PREDICTOR_DELAY mismatch: {cfg.PREDICTOR_DELAY}"
    assert cfg.SMOOTH_WINDOW == 1, f"SMOOTH_WINDOW mismatch: {cfg.SMOOTH_WINDOW}"
    assert cfg.SMOOTH_METHOD == 'mode', f"SMOOTH_METHOD mismatch: {cfg.SMOOTH_METHOD}"
    print(f"✓ USE_GUI: {cfg.USE_GUI}")
    print(f"✓ POLL_SLEEP_DELAY: {cfg.POLL_SLEEP_DELAY}")
    print(f"✓ SMOOTH_METHOD: {cfg.SMOOTH_METHOD}")
    
    # Test computed properties
    print("\n--- Computed Properties ---")
    assert cfg.SESSION_PATH == Path("./Datasets/D1"), f"SESSION_PATH mismatch: {cfg.SESSION_PATH}"
    assert cfg.SAVE_PATH == cfg.SESSION_PATH, f"SAVE_PATH mismatch: {cfg.SAVE_PATH}"
    assert cfg.DATAFOLDER == cfg.SESSION_PATH, f"DATAFOLDER mismatch: {cfg.DATAFOLDER}"
    print(f"✓ SESSION_PATH: {cfg.SESSION_PATH}")
    print(f"✓ SAVE_PATH: {cfg.SAVE_PATH}")
    print(f"✓ MODEL_PATH: {cfg.MODEL_PATH}")
    
    # Test config file metadata
    print("\n--- Config Metadata ---")
    assert cfg.CONFIG_FILE_NAME == "base_config_example.py", f"CONFIG_FILE_NAME mismatch: {cfg.CONFIG_FILE_NAME}"
    print(f"✓ CONFIG_FILE_NAME: {cfg.CONFIG_FILE_NAME}")
    print(f"✓ CONFIG_FILE_PATH: {cfg.CONFIG_FILE_PATH}")
    
    print("\n✅ TEST 1 PASSED: All fields loaded correctly!")
    return cfg


def test_save_and_load_json():
    """Test saving config to JSON and loading it back"""
    
    print("\n" + "="*70)
    print("TEST 2: Save to JSON and Load from JSON")
    print("="*70)
    
    # First load the base config
    config_file = Path(__file__).parent.parent / "config_examples" / "base_config_example.py"
    cfg_original = load_py_config(config_file)
    
    # Save to JSON
    save_dir = Path(__file__).parent.parent / "tests"
    print(f"\n--- Saving config to {save_dir} ---")
    saved_file = save_config(cfg_original, save_dir, name="test_config")
    print(f"✓ Saved to: {saved_file}")
    
    # Load from JSON
    print(f"\n--- Loading config from {saved_file.name} ---")
    cfg_loaded = load_json_config(saved_file)
    
    # Verify core fields match
    print("\n--- Verifying Loaded Data ---")
    assert cfg_loaded.SESSION == cfg_original.SESSION, "SESSION mismatch"
    assert cfg_loaded.SAMPLING == cfg_original.SAMPLING, "SAMPLING mismatch"
    assert cfg_loaded.NUM_CLASSES == cfg_original.NUM_CLASSES, "NUM_CLASSES mismatch"
    assert cfg_loaded.CLASSES == cfg_original.CLASSES, "CLASSES mismatch"
    assert cfg_loaded.MAJORITY_VOTE == cfg_original.MAJORITY_VOTE, "MAJORITY_VOTE mismatch"
    assert cfg_loaded.USE_GUI == cfg_original.USE_GUI, "USE_GUI mismatch"
    assert cfg_loaded.SMOOTH_METHOD == cfg_original.SMOOTH_METHOD, "SMOOTH_METHOD mismatch"
    print(f"✓ SESSION: {cfg_loaded.SESSION}")
    print(f"✓ SAMPLING: {cfg_loaded.SAMPLING}")
    print(f"✓ NUM_CLASSES: {cfg_loaded.NUM_CLASSES}")
    print(f"✓ CLASSES: {cfg_loaded.CLASSES}")
    print(f"✓ SMOOTH_METHOD: {cfg_loaded.SMOOTH_METHOD}")
    
    # Verify computed properties still work
    print("\n--- Verifying Computed Properties ---")
    assert cfg_loaded.SESSION_PATH == Path("./Datasets/D1"), f"SESSION_PATH mismatch: {cfg_loaded.SESSION_PATH}"
    print(f"✓ SESSION_PATH: {cfg_loaded.SESSION_PATH}")
    
    # Verify config file metadata for loaded JSON
    print("\n--- Config Metadata (from JSON) ---")
    assert saved_file.name in cfg_loaded.CONFIG_FILE_NAME, f"CONFIG_FILE_NAME mismatch: {cfg_loaded.CONFIG_FILE_NAME}"
    print(f"✓ CONFIG_FILE_NAME: {cfg_loaded.CONFIG_FILE_NAME}")
    
    print("\n✅ TEST 2 PASSED: JSON save/load works correctly!")
    
    # Cleanup
    print(f"\n--- Cleanup ---")
    saved_file.unlink()
    print(f"✓ Removed test file: {saved_file.name}")


def test_extra_fields():
    """Test that EXTRA fields work correctly"""
    
    print("\n" + "="*70)
    print("TEST 3: EXTRA Fields Functionality")
    print("="*70)
    
    config_file = Path(__file__).parent.parent / "config_examples" / "base_config_example.py"
    cfg = load_py_config(config_file)
    
    # Add custom extra field
    print("\n--- Adding Custom Fields ---")
    cfg.CUSTOM_FIELD = "test_value"
    cfg.CUSTOM_NUMBER = 42
    print(f"✓ Added CUSTOM_FIELD: {cfg.CUSTOM_FIELD}")
    print(f"✓ Added CUSTOM_NUMBER: {cfg.CUSTOM_NUMBER}")
    
    # Verify they're accessible
    assert cfg.CUSTOM_FIELD == "test_value", "CUSTOM_FIELD not accessible"
    assert cfg.CUSTOM_NUMBER == 42, "CUSTOM_NUMBER not accessible"
    assert cfg.get("CUSTOM_FIELD") == "test_value", "get() method failed"
    
    print("\n✅ TEST 3 PASSED: EXTRA fields work correctly!")


def test_load_yaml():
    """Test loading config from YAML file"""
    
    print("\n" + "="*70)
    print("TEST 4: Load from YAML")
    print("="*70)
    
    config_file = Path(__file__).parent.parent / "config_examples" / "base_config_example.yaml"
    
    print(f"\n--- Loading config from {config_file.name} ---")
    cfg = load_yaml_config(config_file)
    
    # Verify core fields
    print("\n--- Verifying Core Fields ---")
    assert cfg.BASE_PATH == Path("./Datasets/"), f"BASE_PATH mismatch: {cfg.BASE_PATH}"
    assert cfg.SESSION == "D1", f"SESSION mismatch: {cfg.SESSION}"
    assert cfg.SAMPLING == 1010, f"SAMPLING mismatch: {cfg.SAMPLING}"
    assert cfg.WINDOW_SIZE == 200, f"WINDOW_SIZE mismatch: {cfg.WINDOW_SIZE}"
    print(f"✓ BASE_PATH: {cfg.BASE_PATH}")
    print(f"✓ SESSION: {cfg.SESSION}")
    print(f"✓ SAMPLING: {cfg.SAMPLING}")
    
    # Verify gesture configuration
    print("\n--- Verifying Gesture Configuration ---")
    assert cfg.NUM_CLASSES == 5, f"NUM_CLASSES mismatch: {cfg.NUM_CLASSES}"
    assert cfg.CLASSES == [2, 3, 30, 14, 18], f"CLASSES mismatch: {cfg.CLASSES}"
    print(f"✓ NUM_CLASSES: {cfg.NUM_CLASSES}")
    print(f"✓ CLASSES: {cfg.CLASSES}")
    
    # Verify controller settings
    print("\n--- Verifying Controller Settings ---")
    assert cfg.SMOOTH_METHOD == 'mode', f"SMOOTH_METHOD mismatch: {cfg.SMOOTH_METHOD}"
    assert cfg.HEARTBEAT_INTERVAL == 0.1, f"HEARTBEAT_INTERVAL mismatch: {cfg.HEARTBEAT_INTERVAL}"
    print(f"✓ SMOOTH_METHOD: {cfg.SMOOTH_METHOD}")
    print(f"✓ HEARTBEAT_INTERVAL: {cfg.HEARTBEAT_INTERVAL}")
    
    # Verify computed properties
    print("\n--- Verifying Computed Properties ---")
    assert cfg.SESSION_PATH == Path("./Datasets/D1"), f"SESSION_PATH mismatch: {cfg.SESSION_PATH}"
    print(f"✓ SESSION_PATH: {cfg.SESSION_PATH}")
    
    # Verify config file metadata
    print("\n--- Config Metadata ---")
    assert cfg.CONFIG_FILE_NAME == "base_config_example.yaml", f"CONFIG_FILE_NAME mismatch: {cfg.CONFIG_FILE_NAME}"
    print(f"✓ CONFIG_FILE_NAME: {cfg.CONFIG_FILE_NAME}")
    
    print("\n✅ TEST 4 PASSED: YAML loading works correctly!")


def run_all_tests():
    """Run all test functions"""
    
    print("\n" + "🧪"*35)
    print("RUNNING ALL CONFIG SYSTEM TESTS")
    print("🧪"*35)
    
    try:
        test_base_example()
        test_save_and_load_json()
        test_extra_fields()
        test_load_yaml()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
