from abc import ABC, abstractmethod


class HandInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def send_gesture(self, gesture):
        pass

    @abstractmethod
    def send_finger_position(self, finger, position):
        pass