"""Centralized argument parser for EMaGerLib examples"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime


def create_parser(description: str, default_config: str = None) -> argparse.ArgumentParser:
    """Create a base argument parser with common options for all examples.
    
    Args:
        description: Description of the script
        default_config: Default configuration file path (relative to project root)
        
    Returns:
        ArgumentParser with common arguments configured
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Configuration file argument
    if default_config is None:
        default_config = str(Path(__file__).parent.parent.parent / "config_examples" / "base_config_example.py")
    
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default=default_config,
        help=f"Path to configuration file to load (default: {default_config})"
    )
    
    # Configuration saving arguments
    parser.add_argument(
        "--save-config-path",
        type=str,
        default=None,
        help="Path to save the loaded configuration file (e.g., ./configs/my_config.json). Overrides config default."
    )
    
    parser.add_argument(
        "--save-config-name",
        type=str,
        default=None,
        help="Custom name for saved config file (auto-generated if not specified). Overrides config default."
    )
    
    parser.add_argument(
        "--save-config-format",
        type=str,
        choices=["json", "yaml"],
        default=None,
        help="Format to save config file (json or yaml). Overrides config default."
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-file-path",
        type=str,
        default=None,
        help="Path to save log file. If not specified, uses config default or console only."
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level. Overrides config default."
    )
    
    parser.add_argument(
        "--log-file-name",
        type=str,
        default=None,
        help="Custom name for log file (auto-generated if not specified). Overrides config default."
    )
    
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        default=None,
        help="Enable file logging. Overrides config default."
    )
    
    parser.add_argument(
        "--no-log-to-file",
        action="store_true",
        default=None,
        help="Disable file logging. Overrides config default."
    )
    
    return parser


def setup_logging(args: argparse.Namespace, cfg = None, script_name: str = None) -> None:
    """Configure logging based on parsed arguments with config as fallback.
    
    Args:
        args: Parsed arguments from argparse
        cfg: Configuration object (CoreConfig) - used as fallback for unset args
        script_name: Name of the script (used for default log filename)
    """
    # Determine log level (args override config)
    if args.log_level is not None:
        log_level_str = args.log_level
    elif cfg is not None and hasattr(cfg, 'LOG_LEVEL'):
        log_level_str = cfg.LOG_LEVEL
    else:
        log_level_str = "INFO"
    log_level = getattr(logging, log_level_str)
    
    # Format for log messages
    log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure handlers
    handlers = []
    
    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)
    
    # Determine if file logging is enabled
    log_to_file = False
    if args.log_to_file:
        log_to_file = True
    elif args.no_log_to_file:
        log_to_file = False
    elif args.log_file_path or args.log_file_name:
        log_to_file = True
    elif cfg is not None and hasattr(cfg, 'LOG_TO_FILE'):
        log_to_file = cfg.LOG_TO_FILE
    
    # File handler (if enabled)
    if log_to_file:
        # Determine log file path (args override config)
        if args.log_file_path:
            log_path = Path(args.log_file_path)
        else:
            # Use config or auto-generate
            if cfg is not None and hasattr(cfg, 'LOG_FILE_PATH') and cfg.LOG_FILE_PATH:
                log_path = Path(cfg.LOG_FILE_PATH)
            else:
                # Auto-generate log file name
                if script_name is None:
                    script_name = "emager"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Determine log filename
                if args.log_file_name:
                    log_filename = args.log_file_name
                elif cfg is not None and hasattr(cfg, 'LOG_FILE_NAME') and cfg.LOG_FILE_NAME:
                    log_filename = cfg.LOG_FILE_NAME
                else:
                    log_filename = f"{script_name}_{timestamp}.log"
                
                # Default to logs directory in current working directory
                log_dir = Path.cwd() / "logs"
                log_dir.mkdir(exist_ok=True)
                log_path = log_dir / log_filename
        
        # Create file handler
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)
        
        # Keep this print for user visibility about log file location
        print(f"Logging to file: {log_path}")
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logging.info(f"Logging initialized at {log_level_str} level")


def save_config_if_requested(args: argparse.Namespace, cfg, script_name: str = None) -> None:
    """Save configuration file if requested via command-line arguments or config defaults.
    
    Args:
        args: Parsed arguments from argparse
        cfg: Configuration object (CoreConfig)
        script_name: Name of the script (used for default config filename)
    """
    from emagerlib.config.save_config import save_config
    
    # Check if saving is requested via args or config defaults
    should_save = (args.save_config_path or args.save_config_name or 
                   (hasattr(cfg, 'SAVE_CONFIG_PATH') and cfg.SAVE_CONFIG_PATH) or
                   (hasattr(cfg, 'SAVE_CONFIG_NAME') and cfg.SAVE_CONFIG_NAME))
    
    if should_save:
        # Determine save path (args override config)
        if args.save_config_path:
            save_path = Path(args.save_config_path)
            save_dir = save_path.parent
            # If full path provided, extract name and use provided extension
            if save_path.suffix:
                config_name = save_path.stem
            else:
                # If directory only, use config_name or auto-generate
                config_name = args.save_config_name if args.save_config_name else f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        elif hasattr(cfg, 'SAVE_CONFIG_PATH') and cfg.SAVE_CONFIG_PATH:
            save_path = Path(cfg.SAVE_CONFIG_PATH)
            save_dir = save_path.parent
            if save_path.suffix:
                config_name = save_path.stem
            else:
                config_name = args.save_config_name if args.save_config_name else (cfg.SAVE_CONFIG_NAME if hasattr(cfg, 'SAVE_CONFIG_NAME') and cfg.SAVE_CONFIG_NAME else f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            # Auto-generate path
            save_dir = Path.cwd() / "saved_configs"
            config_name = args.save_config_name if args.save_config_name else (cfg.SAVE_CONFIG_NAME if hasattr(cfg, 'SAVE_CONFIG_NAME') and cfg.SAVE_CONFIG_NAME else f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Determine format (args override config)
        if args.save_config_format:
            file_format = args.save_config_format
        elif hasattr(cfg, 'SAVE_CONFIG_FORMAT'):
            file_format = cfg.SAVE_CONFIG_FORMAT
        else:
            file_format = "json"
        
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        saved_file = save_config(cfg, save_dir, name=config_name, file_format=file_format)
        logging.info(f"Configuration saved to: {saved_file}")
        # Keep this print for user visibility about saved config location
        print(f"[SAVED] Configuration saved to: {saved_file}")
