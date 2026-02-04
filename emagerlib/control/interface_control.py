# from control.zeus_control import ZeusControl
# from control.smart_hand_control import SmartHandControl
from src.control.psyonic_control import PsyonicHandControl

class InterfaceControl:
    def __init__(self, hand_type, **kwargs):
        """
        Initialize the interface control with the specified hand type.
        
        Args:
            hand_type (str): Type of hand to use ("zeus" or "smart" or "psyonic")
            **kwargs: Additional arguments to pass to the hand controller
        """
        self.hand_type = hand_type.lower()
        self.hand = None
        self.kwargs = kwargs
        
        self.initialize_hand()

    def initialize_hand(self):
        """Initialize the appropriate hand controller based on the hand type."""
        # if self.hand_type == "zeus":
        #     self.hand = ZeusControl(**self.kwargs)
        # elif self.hand_type == "smart":
        #     self.hand = SmartHandControl(**self.kwargs)
        if self.hand_type == "psyonic":
            self.hand = PsyonicHandControl(**self.kwargs)
        else:
            raise ValueError(f"Unsupported hand type: {self.hand_type}")

    def connect(self):
        """Connect to the hand device."""
        if not self.hand:
            self.initialize_hand()
        self.hand.connect()

    def disconnect(self):
        """Disconnect from the hand device."""
        if self.hand:
            self.hand.disconnect()

    def send_gesture(self, gesture, direct=False):
        """Send a gesture to the hand."""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        if direct:
            if hasattr(self.hand, 'send_gesture_direct'):
                self.hand.send_gesture_direct(gesture)
            else:
                raise NotImplementedError(f"send_gesture_direct not implemented for {self.hand_type} hand")
        else:
            self.hand.send_gesture(gesture)

    def send_finger_position(self, finger, position):
        """Send a specific finger position to the hand. finger is 0-4, position is 0-100"""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        self.hand.send_finger_position(finger, position)

    def read_data(self):
        """Read data from the hand device."""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        self.hand.read_data()

    def send_data(self, data, data_id=None):
        """Send raw data to the hand device."""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        if hasattr(self.hand, 'send_data'):
            if data_id is not None and hasattr(self.hand, 'send_data_with_id'):
                self.hand.send_data_with_id(data, data_id)
            else:
                self.hand.send_data(data)
        else:
            raise NotImplementedError(f"send_data not implemented for {self.hand_type} hand")

    def start_telemetry(self):
        """Start telemetry if supported by the hand."""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        if hasattr(self.hand, 'start_telemetry'):
            self.hand.start_telemetry()
        else:
            raise NotImplementedError(f"Telemetry not supported for {self.hand_type} hand")

    def stop_telemetry(self):
        """Stop telemetry if supported by the hand."""
        if not self.hand:
            raise RuntimeError("Hand not initialized. Call connect() first.")
        if hasattr(self.hand, 'stop_telemetry'):
            self.hand.stop_telemetry()
        else:
            raise NotImplementedError(f"Telemetry not supported for {self.hand_type} hand") 