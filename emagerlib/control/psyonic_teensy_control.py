"""
Psyonic Hand Control via Teensy UART Controller

This module provides an interface to control the Psyonic hand through a Teensy
microcontroller using UART serial communication. The Teensy acts as an intermediary,
handling continuous updates to the hand at high frequency.
"""

from emagerlib.control.abstract_hand_control import HandInterface
from emagerlib.control.gesture_decoder import decode_gesture
from emagerlib.utils.find_usb import find_teensy    
import serial
import serial.tools.list_ports
import time
import threading
from typing import List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class PsyonicTeensyController:
    """
    Low-level controller class for Teensy-based Psyonic Hand system.
    Provides methods for each available command.
    
    Note: Commands are case-insensitive. Methods use short commands internally.
    """
    
    def __init__(self, port: str, baudrate: int = 115200, auto_read: bool = True, 
                 response_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the controller.
        
        Args:
            port: Serial port (e.g., 'COM3', '/dev/ttyACM0')
            baudrate: Baud rate (default: 115200)
            auto_read: Automatically read responses in background thread
            response_callback: Optional callback function for responses (receives line as string)
        """
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.response_callback = response_callback
        self._auto_read = auto_read
        self._running = False
        self._read_thread = None
        
        # Wait for connection to establish
        time.sleep(2.5)
        
        if auto_read:
            self.start_reading()
    
    def start_reading(self):
        """Start background thread to read responses"""
        if not self._running:
            self._running = True
            self._read_thread = threading.Thread(target=self._read_responses, daemon=True)
            self._read_thread.start()
    
    def stop_reading(self):
        """Stop background reading thread"""
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=1)
    
    def _read_responses(self):
        """Background thread to continuously read responses"""
        while self._running:
            if self.ser.in_waiting:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').rstrip()
                    if line:
                        if self.response_callback:
                            self.response_callback(line)
                        else:
                            logger.debug(f"[Teensy] {line}")
                except Exception as e:
                    logger.error(f"Error reading response: {e}")
            time.sleep(0.01)
    
    def _send_command(self, cmd: str):
        """
        Internal method to send a command.
        
        Args:
            cmd: Command string to send (can include parameters)
        """
        self.ser.write(f"{cmd}\n".encode())
    
    def set_target_positions(self, positions: List[int]):
        """
        Set target positions for all 6 fingers.
        
        Args:
            positions: List of 6 integer positions (degrees)
                      Fingers 0-4 (thumb flex, index, middle, ring, pinky): 0-150 degrees
                      Finger 5 (thumb rotation): -100 to +100 degrees
            
        Example:
            controller.set_target_positions([10, 20, 30, 40, 50, 60])
        """
        if len(positions) != 6:
            raise ValueError("Must provide exactly 6 position values")
        
        # Validate position ranges
        for i, pos in enumerate(positions):
            if i < 5:  # Fingers 0-4 (thumb flex through pinky)
                if not (0 <= pos <= 150):
                    raise ValueError(f"Position {i} must be 0-150 degrees, got {pos}")
            else:  # Finger 5 (thumb rotation)
                if not (-100 <= pos <= 100):
                    raise ValueError(f"Position 5 (thumb rotation) must be -100 to +100 degrees, got {pos}")
        
        position_str = ' '.join(str(p) for p in positions)
        self._send_command(f'p {position_str}')
    
    def toggle_hand_thread(self):
        """Start or stop the hand communication thread"""
        self._send_command('t')
    
    def get_status(self):
        """Print system status"""
        self._send_command('s')
    
    def close(self):
        """Close the serial connection"""
        self.stop_reading()
        if self.ser.is_open:
            self.ser.close()
    
    def is_connected(self) -> bool:
        """Check if serial connection is open"""
        return self.ser.is_open
    
    @staticmethod
    def find_teensy_port() -> Optional[str]:
        """
        Attempt to automatically find Teensy COM port.
        
        Returns:
            Port name if found, None otherwise
        """
        teensy_port = find_teensy()
        
        return teensy_port
    
    @staticmethod
    def list_available_ports():
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        logger.info("Available serial ports:")
        for i, port in enumerate(ports):
            logger.info(f"  [{i}] {port.device}: {port.description}")
        return [port.device for port in ports]


class PsyonicTeensyControl(HandInterface):
    """
    High-level interface for Psyonic hand control via Teensy.
    Implements the HandInterface protocol for integration with EMaGerLib.
    """
    
    def __init__(self, port: str = None, baudrate: int = 115200, **kwargs):
        """
        Initialize the Psyonic Teensy hand controller.
        
        Args:
            port: Serial port to connect to (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux)
                  If None, will auto-detect Teensy
            baudrate: Serial communication baudrate (default: 115200)
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.port = port
        self.baudrate = baudrate
        self.teensy = None
        self.connected = False
        self._current_positions = [0, 0, 0, 0, 0, 0]  # Track current positions

    def connect(self):
        """Connect to the Psyonic hand via Teensy."""
        if self.connected:
            return
        
        try:
            # Auto-detect port if not specified
            if self.port is None:
                self.port = PsyonicTeensyController.find_teensy_port()
                if self.port is None:
                    logger.error("Could not find Teensy port. Available ports:")
                    PsyonicTeensyController.list_available_ports()
                    raise RuntimeError("Teensy not found. Please specify port manually.")
            
            logger.info(f"Connecting to Teensy on {self.port}...")
            
            # Create controller instance
            self.teensy = PsyonicTeensyController(
                port=self.port,
                baudrate=self.baudrate,
                auto_read=True,
                response_callback=self._handle_response
            )
            
            self.connected = True
            logger.info(f"Connected to Teensy on {self.port}")
            
            # Start the hand thread on the Teensy
            logger.info("Starting Teensy hand communication thread...")
            self.teensy.toggle_hand_thread()
            
            # Get status
            self.teensy.get_status()
            
        except Exception as e:
            logger.error(f"Error connecting to Teensy: {e}")
            self.connected = False
            raise

    def disconnect(self):
        """Disconnect from the Psyonic hand."""
        if self.teensy:
            logger.info("Stopping Teensy hand thread...")
            try:
                self.teensy.toggle_hand_thread()
            except:
                pass
            
            self.teensy.close()
            self.connected = False
            logger.info("Disconnected from Teensy")

    def _handle_response(self, line: str):
        """Handle responses from Teensy."""
        logger.debug(f"[Teensy] {line}")

    def send_gesture(self, gesture: str):
        """
        Send a gesture command to the Psyonic hand.
        
        Args:
            gesture: Name of the gesture to perform
        """
        if not self.connected:
            raise RuntimeError("Not connected to Teensy")
        
        try:
            # Get finger positions from gesture decoder
            positions = decode_gesture(gesture)
            
            # Send positions to Teensy
            self.teensy.set_target_positions(positions)
            self._current_positions = positions
            
            logger.info(f"Sent gesture [{gesture}] -> positions {positions}")
            
        except Exception as e:
            logger.error(f"Error sending gesture: {e}")
            raise

    def send_finger_position(self, finger: int, position: int):
        """
        Send a specific finger position to the Psyonic hand.
        
        Args:
            finger: Finger index (0-5)
                    0: Thumb flex
                    1: Index
                    2: Middle
                    3: Ring
                    4: Pinky
                    5: Thumb rotation
            position: Position value (0-100 for fingers 0-4, -100 to 100 for finger 5)
        """
        if not self.connected:
            raise RuntimeError("Not connected to Teensy")
        
        if not 0 <= finger <= 5:
            raise ValueError("Finger index must be between 0 and 5")
        
        # Update the position in our tracking array
        self._current_positions[finger] = position
        
        # Send all positions (Teensy requires all 6 values)
        try:
            self.teensy.set_target_positions(self._current_positions)
            logger.debug(f"Set finger {finger} to position {position}")
        except Exception as e:
            logger.error(f"Error setting finger position: {e}")
            raise
    
    def send_finger_positions(self, positions: List[int]):
        """
        Send multiple finger positions at once.
        
        Args:
            positions: List of 6 position values (same format as set_target_positions)
        """
        if not self.connected:
            raise RuntimeError("Not connected to Teensy")
        
        try:
            self.teensy.set_target_positions(positions)
            self._current_positions = positions
            logger.debug(f"Set multiple finger positions: {positions}")
        except Exception as e:
            logger.error(f"Error setting finger positions: {e}")
            raise
