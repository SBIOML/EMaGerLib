"""
EMaGerLib Control Module

This module provides interfaces for controlling various prosthetic hands.
"""

from emagerlib.control.abstract_hand_control import HandInterface
from emagerlib.control.interface_control import InterfaceControl
from emagerlib.control.psyonic_control import PsyonicHandControl
from emagerlib.control.psyonic_teensy_control import PsyonicTeensyControl, PsyonicTeensyController

__all__ = [
    'HandInterface',
    'InterfaceControl',
    'PsyonicHandControl',
    'PsyonicTeensyControl',
    'PsyonicTeensyController',
]
