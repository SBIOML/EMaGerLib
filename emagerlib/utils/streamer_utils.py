"""
Utility functions for EMaGer streamer version management.
"""
import logging
from typing import Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def get_emager_streamer(version: str = "v3.0") -> Tuple[Any, Any]:
    """
    Get the appropriate EMaGer streamer based on the device version.
    
    Args:
        version: EMaGer device version ("v1.0", "v1.1", or "v3.0")
        
    Returns:
        Tuple of (process, shared_memory_items)
        
    Raises:
        ValueError: If the version is not supported
    """
    
    
    # Normalize version string
    version = version.lower().strip()
    if version in ["v1.0", "v1", "1.0", "1"]:
        logger.info("Using EMaGer v1.0 streamer")
        from libemg.streamers import emager_streamer
        return emager_streamer()
    elif version in ["v1.1", "1.1"]:
        logger.info("Using EMaGer v1.1 streamer")
        from libemg.streamers import emager_streamer
        return emager_streamer(version="v1.1")
    elif version in ["v3.0", "v3", "3.0", "3"]:
        logger.info("Using EMaGer v3 streamer")
        from libemg.streamers import emagerv3_streamer
        return emagerv3_streamer()
    else:
        raise ValueError(
            f"Unsupported EMaGer version: {version}. "
            f"Supported versions are 'v1.0', 'v1.1', or 'v3.0'"
        )
