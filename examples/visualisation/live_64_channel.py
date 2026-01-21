import time
import numpy as np
import serial
from scipy import signal
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import QtGui
import serial.tools.list_ports


def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    number_of_packet = int(len(data) / 128)
    roll_data = []
    for i in range(number_of_packet):
        data_lsb = data[i * 128:(i + 1) * 128] & np.ones(128, dtype=np.int8)
        mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
        try:
            offset = np.where(mask_match == match_result)[0][0] - 3
        except IndexError:
            return None
        roll_data.append(np.roll(data[i * 128:(i + 1) * 128], -offset))
    return roll_data


class HDSensor(object):
    '''
    Sensor object for data logging from HD EMG sensor
    '''

    def __init__(self, serialpath, BR):
        print('init sensor')
        '''
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param serialpath: (str) - Path to serial port
        :param BR: (int) - Com port baudrate
        '''
        self.ser = serial.Serial(serialpath, BR, timeout=1)
        self.ser.close()

        self.bytes_to_read = 128
        ### ^ Number of bytes in message (i.e. channel bytes + header/tail bytes)
        self.mask = np.array([0, 2] + [0, 1] * 63)
        ### ^ Template mask for template matching on input data
        # self.channelMap = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
        #                   [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
        #                   [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
        #                   [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
                          
        self.channelMap = [44, 49, 43, 55, 39, 59, 33, 2, 32, 3, 26, 6, 22, 13, 16, 10] + \
                      [42, 48, 45, 54, 38, 58, 35, 0, 34, 1, 27, 7, 23, 11, 17, 12] + \
                      [46, 52, 40, 51, 36, 56, 31, 60, 30, 63, 25, 4, 21, 8, 18, 15] + \
                      [47, 50, 41, 53, 37, 57, 29, 62, 28, 61, 24, 5, 19, 9, 20, 14]

        # [i for i in range(27, 32)] + [0, 1, 2] + [i for i in range(23, 27)] + \
        #                [i for i in range(3, 7)] + [22, 21, 20, 19] + [10, 9, 8, 7] + [18, 17, 16, 15, 14, 13, 12, 11]
        #                 ### ^ Channel map to hardware sensor obtained from lab tests, needed to reorder channels

    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        self.ser.reset_input_buffer()
        return

    def close(self):
        self.ser.close()
        return

    def open(self):
        self.ser.open()
        return

    def read(self, readtime, feedback=False, savetxt=False, savepath=None):
        '''
        Read the incoming data in com port for a given time.
        :param readtime: (int) - reading time period (seconds)
        :param feedback: (bool) - print notice upon receiving corrupted data
        :param savetxt: (bool) - save read data to csv
        :param savepath: (str) - path for saved data
        :return: (list of lists) - list of channels' listed data points (e.g. 64xN for 64 channels of N data points)
        '''
        data = [[] for i in range(64)]

        ### ^ 64 = n_channels
        self.open()
        self.clear_buffer()

        start_time = time.time()
        while (time.time() - start_time) < readtime:
            data_packet = reorder(list(self.ser.read(self.bytes_to_read)), self.mask, 63)
            if data_packet is not None:
                samples = [int.from_bytes(bytes([data_packet[i * 2], data_packet[i * 2 + 1]]), 'big', signed=True) for i
                           in range(64)]
                ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.
                for i, d in enumerate(data):
                    d += [samples[i]]
                    ### ^ Separating recorded data to respective channels
            elif feedback:
                print('Corrupted data. Dropped packet.')
            else:
                pass
        self.close()
        data_remap = []
        for i in self.channelMap:
            data_remap += [data[i]]
        #                 ### ^ Remapping data channels

        if savetxt:
            np.savetxt(savepath, data_remap, delimiter=',', fmt='%s')

        return data_remap  # data_remap

    def sample(self):
        '''
        Sample 1 message from com port (1 sample from each channel), retry until valid reception in case of
        corrupted data.
        :return: (list) - containing the 64 samples (1 for each channel)
        '''
        # self.open()
        self.clear_buffer()
        while (True):
            data_packet = reorder(list(self.ser.read(128)), self.mask, 63)
            if data_packet is not None:
                sample = [int.from_bytes(bytes([data_packet[i * 2], data_packet[i * 2 + 1]]), 'big', signed=True) for i
                          in
                          range(64)]  ### ^ Iterating over byte pairs in line, 32 => n_channels, 2 bytes per ch.
                # sample = [sample[i] for i in self.channelMap]
                #                             ### ^ Remapping data channels
                # self.close()
                return sample

    def live_read(self, feedback=False, savetxt=False, savepath=None, firstTime=False, decimate=False):
        '''
        Read the incoming data in com port for a given time.
        :param readtime: (int) - reading time period (seconds)
        :param feedback: (bool) - print notice upon receiving corrupted data
        :param savetxt: (bool) - save read data to csv
        :param savepath: (str) - path for saved data
        :return: (list of lists) - list of channels' listed data points (e.g. 64xN for 64 channels of N data points)
        '''
        data = [[] for i in range(64)]
        ### ^ 64 = n_channels
        if firstTime:
            self.open()
            self.clear_buffer()
            time.sleep(0.0005)

        data_packet = None
        while data_packet is None:
            bytes_available = self.ser.inWaiting()
            bytesToRead = bytes_available - (bytes_available % 128)
            data_packet = reorder(list(self.ser.read(bytesToRead)), self.mask, 63)
        for packet in data_packet:
            samples = [int.from_bytes(bytes([packet[i * 2], packet[i * 2 + 1]]), 'big', signed=True) for i in range(64)]
            for i, d in enumerate(data):
                d += [samples[i]]
        ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.
        ### ^ Separating recorded data to respective channels
        data_remap = []
        if not decimate:
            for i in self.channelMap:
                data_remap += [data[i]]
        #                 ### ^ Remapping data channels
        else:
            for i in self.channelMap:
                print(len(data[i]))
                data_remap += [signal.decimate(data[i], 2)]
        # data_remap = signal.decimate(data_remap, 2)
        # print("before:", len(data_remap[0]))
        nb_pts = len(data_remap[0])
        # print(len(data_remap[0]))
        return data_remap, nb_pts  # data_remap

    def read_full_buffer(self, feedback=False, savetxt=False, savepath=None):
        '''
        Read the incoming data in com port for a given time.
        :param readtime: (int) - reading time period (seconds)
        :param feedback: (bool) - print notice upon receiving corrupted data
        :param savetxt: (bool) - save read data to csv
        :param savepath: (str) - path for saved data
        :return: (list of lists) - list of channels' listed data points (e.g. 64xN for 64 channels of N data points)
        '''
        # Receives data
        data = [[] for i in range(64)]

        bytes_available = 0
        data_packet = None
        while data_packet is None:
            while bytes_available < 1024:
                bytes_available = self.ser.inWaiting()
            # print("bytes available:", bytes_available)
            bytesToRead = bytes_available - (bytes_available % 128)
            data_packet = reorder(list(self.ser.read(bytesToRead)), self.mask, 63)

        for packet in data_packet:
            samples = [int.from_bytes(bytes([packet[i * 2], packet[i * 2 + 1]]), 'big', signed=True) for i in range(64)]
            for i, d in enumerate(data):
                d += [samples[i]]
        ### ^ Iterating over byte pairs in line, 64 => n_channels, 2 bytes per ch.
        ### ^ Separating recorded data to respective channels
        data_remap = []
        for i in self.channelMap:
            data_remap += [data[i]]
        #                 ### ^ Remapping data channels
        # print("Out of func", np.transpose(np.array(data_remap)).dtype)
        # print(len(data_remap[0]))
        return np.transpose(np.array(data_remap))  # data_remap
class RealTimeOscilloscope:
    def __init__(self, num_signals, data_points, refresh_rate, sensor):
        self.num_signals = num_signals
        self.data_points = data_points
        self.refresh_rate = refresh_rate
        self.sensor = sensor
        self.firstGo = True

        # Time axis
        self.t = np.linspace(0, 3, data_points)

        # Create the application
        self.app = QApplication([])

        # Create a window
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Real-Time Oscilloscope')
        self.win.setBackground(QtGui.QColor(255, 255, 255))  # White background
        self.win.show()

        # Initialize the data buffer
        self.data = [np.zeros(data_points) for _ in range(num_signals)]

        # Plot layout
        num_rows = 16
        num_columns = (num_signals + num_rows - 1) // num_rows  # ceil(num_signals / num_rows)

        self.plots = []
        for i in range(num_signals):
            row = i % num_rows
            col = i // num_rows

            p = self.win.addPlot(row=row, col=col)
            p.setYRange(-10000, 10000)
            p.getAxis('left').setStyle(showValues=False)
            p.getAxis('bottom').setStyle(showValues=False)

            graph = p.plot(self.t, self.data[i], pen=pg.mkPen(color='r', width=2))
            self.plots.append(graph)

        # Timer
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000 // refresh_rate)

    def update(self):
        new_data, nb_pts = self.sensor.live_read(firstTime=self.firstGo, decimate=False)
        self.firstGo = False

        if nb_pts > 0:
            for i in range(self.num_signals):
                self.data[i] = np.roll(self.data[i], -nb_pts)
                self.data[i][-nb_pts:] = new_data[i]
                self.plots[i].setData(self.t, self.data[i])

    def run(self):
        self.app.exec()

def find_port(vid, pid):
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if port.vid == vid and port.pid == pid:
            print(f"Found device: {port.device}")
            return port.device
    raise ValueError("Device not found")


def main():
    print('start')
    PORT = find_port(0x04b4, 0xf155)
    sensor = HDSensor(PORT, 1500000)
    print('sensor init')
    num_signals = 64
    data_points = 3000  # 3 seconds at 100 samples per second
    refresh_rate = 30  # 30Hz refresh rate
    oscilloscope = RealTimeOscilloscope(num_signals, data_points, refresh_rate, sensor)
    oscilloscope.run()

if __name__ == '__main__':
    main()
