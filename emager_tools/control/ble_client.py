import asyncio
from bleak import BleakClient, BleakScanner


class BLEDevice:
    def __init__(self, address):
        self.address = address
        self.client = None
        self.services = {}
        self.notify_callbacks = None

    def _add_service(self, service_uuid):
        self.services[service_uuid] = {}

    def add_characteristic(self, service_uuid, char_uuid, initial_value=None):
        if service_uuid not in self.services:
            self._add_service(service_uuid)
        self.services[service_uuid][char_uuid] = initial_value

    def connect(self):
        self.client = BleakClient(self.address)
        print(f"Connecting to {self.address}")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.connect())

    def disconnect(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.disconnect())
        print(f"Disconnected from {self.address}")
        self.client = None

    def read(self, service_uuid, char_uuid):
        loop = asyncio.get_event_loop()
        value = loop.run_until_complete(self.client.read_gatt_char(char_uuid))
        self.services[service_uuid][char_uuid] = value
        return value

    def write(self, service_uuid, char_uuid, data):
        if isinstance(data, str):
            data = data.encode()
        self.services[service_uuid][char_uuid] = data
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.write_gatt_char(char_uuid, bytearray(data)))

    def start_notify(self, char_uuid):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.start_notify(char_uuid, self._notification_handler))

    def stop_notify(self, char_uuid):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.stop_notify(char_uuid))

    def add_notification_callback(self, callback, args=None):
        '''
        Callback should be a function that takes 3 arguments: sender, data, args
        '''
        self.notify_callbacks = callback
        self.notify_args = args

    def _notification_handler(self, sender, data):
        if self.notify_callbacks:
            self.notify_callbacks(sender, data, self.notify_args)

def scan_and_connect(device_name, retry = 1) -> BLEDevice:
    target_device = None
    for i in range(retry):
        print(f"Scanning for {device_name}... Attempt {i+1}")
        loop = asyncio.get_event_loop()
        devices = loop.run_until_complete(BleakScanner.discover())
        for device in devices:
            if device.name == device_name:
                target_device = device
                ble_device = BLEDevice(target_device)
                ble_device.connect()
                return ble_device
    
    if target_device is None:
        print(f"Device with name {device_name} not found.")
        return None