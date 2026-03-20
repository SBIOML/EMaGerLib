"""Main entry point for EMaGerLib commands.

This script provides a unified interface to all EMaGer commands.
Usage: emager <command> [options]

Example:
    emager train-cnn --config my_config.py
    emager screen-training
    emager --help
"""

import sys
from typing import Dict, Tuple
from emagerlib.utils.arg_parser import create_parser


# Map command names to their module paths and main function names
# Using lazy imports to avoid loading all modules at startup
COMMANDS: Dict[str, Tuple[str, str]] = {
    # Data collection
    "screen-training": ("examples.data_collection.screen_guided_trainning", "main"),
    
    # Hand control
    "test-hand": ("examples.hand_control.test_hand_control", "main"),
    "test-wave": ("examples.hand_control.test_hand_wave", "main"),
    "test-psyonic": ("examples.hand_control.test_psyonic_hand", "main"),
    "test-teensy": ("examples.hand_control.test_teensy_hand", "main"),
    
    # Realtime
    "realtime-control": ("examples.realtime.realtime_control", "main"),
    "realtime-control-teensy": ("examples.realtime.realtime_control_teensy", "main"),
    "realtime-control-threads": ("examples.realtime.realtime_control_threads", "main"),
    "realtime-predict": ("examples.realtime.realtime_prediction", "main"),
    
    # Training
    "train-cnn": ("examples.training.train_cnn", "main"),
    
    # Visualization
    "visualize-libemg": ("examples.visualisation.libemg_visualize", "main"),
    "live-64ch": ("examples.visualisation.live_64_channel", "main"),
    
    # Tests
    "run-tests": ("tests.run_all_tests", "main"),
}


def print_help():
    """Print help message showing available commands."""
    help_text = """
EMaGerLib - Unified command-line interface for EMG gesture recognition

Usage: emager <command> [options]

Available commands:

  Data Collection:
    screen-training     Screen-guided EMG training data collection
  
  Hand Control:
    test-hand          Test hand control interface
    test-wave          Test hand wave gestures
    test-psyonic       Test Psyonic hand control
    test-teensy        Test Teensy-based Psyonic hand control
  
  Real-time:
    realtime-control   Real-time prosthetic hand control
    realtime-control-teensy  Real-time control via Teensy bridge
    realtime-control-threads Real-time threaded shared-memory control
    realtime-predict   Real-time gesture prediction
  
  Training:
    train-cnn          Train CNN model on EMG data
  
  Visualization:
    visualize-libemg   Visualize with libemg tools
    live-64ch          Live 64-channel EMG visualization
  
  Testing:
    run-tests          Run complete test suite

Examples:
  emager train-cnn --config my_config.py
  emager screen-training
  emager realtime-predict --help

For help on a specific command:
  emager <command> --help
"""
    print(help_text)
    
    # Also show common options available to all commands
    print("\nCommon options available to all commands:\n")
    
    parser = create_parser(description="")
    parser.prog = "emager <command>"
    parser.print_help()
    print()


def main():
    """Main entry point for the emager command."""
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        print_help()
        sys.exit(0)
    
    # Check for help flag
    if sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        sys.exit(0)
    
    # Get the command
    command = sys.argv[1]
    
    # Check if command exists
    if command not in COMMANDS:
        print(f"Error: Unknown command '{command}'")
        print(f"\nRun 'emager --help' to see available commands.")
        sys.exit(1)
    
    # Check for help flag early to avoid loading modules unnecessarily
    if any(arg in ['-h', '--help'] for arg in sys.argv[2:]):
        # Create a parser with the standard common arguments
        parser = create_parser(
            description=f"Run {command} command"
        )
        parser.prog = f"emager {command}"
        
        # Display help and exit
        parser.print_help()
        sys.exit(0)
    
    # Get module and function name
    module_path, func_name = COMMANDS[command]
    
    # Remove the command from sys.argv so the sub-command gets clean arguments
    sys.argv = [f"emager {command}"] + sys.argv[2:]
    
    # Dynamically import and execute the command
    try:
        import importlib
        module = importlib.import_module(module_path)
        main_func = getattr(module, func_name)
        main_func()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error executing command '{command}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
