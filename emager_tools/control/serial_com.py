import serial
import serial.tools.list_ports
from time import sleep, time

class SerialCommunication:
    # Common device IDs
    DEVICE_IDS = {
        "TTL": (0x10C4, 0xEA60),
        "Pico": (0x2e8a, 0x0005),
        # "PSOC": (0x04b4, 0xf155),
    }

    def __init__(self, serial=None, port=None, baud_rate=115200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1.5):
        self.port = port
        self.baud_rate = baud_rate
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.serial = serial
        self.device_name = None

    def _find_port(self):
        """Find a device port."""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            if port.vid is not None and port.pid is not None:
                for _device_name, (vid, pid) in self.DEVICE_IDS.items():
                    if self.device_name is not None: 
                        (vid, pid) = self.DEVICE_IDS[self.device_name]
                        _device_name = self.device_name
                    if port.vid == vid and port.pid == pid:
                        print(f"Found device: {_device_name} as {port.device}")
                        print(f"  Manufacturer: {port.manufacturer}")
                        print(f"  Product: {port.product}")
                        return port.device, _device_name
                    if self.device_name is not None:
                        break
        raise ValueError("No device found. Please check connections and specify port manually.")
    
    def open(self):
        if self.port is None:
            self.port, self.device_name = self._find_port()
        if self.serial is None:
            self.serial = serial.Serial(port=self.port, baudrate=self.baud_rate, parity=self.parity, stopbits=self.stopbits, timeout=self.timeout)
        if not self.serial.is_open:
            self.serial.open()
        sleep(0.2)

    def close(self):
        if self.serial is not None:
            if self.serial.is_open:
                self.serial.close()

    def write(self, message):
        if isinstance(message, str):
            message = (message + "\n").encode()
        elif isinstance(message, bytes) or isinstance(message, bytearray):
            message = bytes(message) + b'\n'
        if self.serial is not None:
            self.serial.write(message)
            self.read()

    def read(self):
        if self.serial is not None:
            return self.serial.read_until().strip()

    def test(self):
        """Test serial communication by sending and reading data."""
        try:
            print("Testing serial communication...")
            
            # Try to open connection
            print("Opening serial port...")
            self.open()
            print(f"Connected to device {self.device_name} as {self.port} at {self.baud_rate} baud")
            
            # Send test message
            test_msg = "TEST"
            print(f"\nSending test message: {test_msg}")
            self.write(test_msg)
            
            # Read response
            response = self.read()
            print(f"Received response: {response}")
            
            # Close connection
            print("\nClosing serial port...")
            self.close()
            print("Test completed successfully")
            
        except Exception as e:
            print(f"Error during serial test: {e}")
            if self.serial and self.serial.is_open:
                self.close()

if __name__ == "__main__":
    serial_com = SerialCommunication()
    serial_com.test()

