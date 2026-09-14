#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live 6-axis IMU in 3D space, with data coming from LibEMG shared memory
via your EmagerV3Streamer (buffer_write via vstack prepend).

"""

import time
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMainWindow
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QMatrix4x4
import trimesh
import pyqtgraph.opengl as gl
from libemg.shared_memory_manager import SharedMemoryManager
import time
from imufusion import Ahrs, AhrsSettings, CONVENTION_NWU


# -----------------------------
# Shared memory modality names
# -----------------------------
MOD_IMU = "imu"
MOD_IMU_COUNT = "imu_count"


def load_stl(path):
    mesh = trimesh.load_mesh(path)
    mesh.fix_normals()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    mesh_data = gl.MeshData(
        vertexes=vertices,
        faces=faces
    )

    return gl.GLMeshItem(
        meshdata=mesh_data,
        smooth=False,
        drawFaces=True,
        drawEdges=False,
        shader='normalColor',
    )

class GUI(QWidget): 
    def __init__(self, smi): 
        self.app = QApplication([]) 
        self.smi = smi

        super().__init__() 

        lay = QVBoxLayout(self)

        self.setWindowTitle("IMU Viewer")

        self.gizmo = IMUGizmo()
        lay.addWidget(self.gizmo)

        self.calibrating = False

        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.start_calibration)
        lay.addWidget(self.calibrate_button)

        self.calibration_timer = QTimer(self)
        self.calibration_timer.setSingleShot(True)
        self.calibration_timer.timeout.connect(self.end_calibration)

        self.worker = GestureWorker(self)
        self.worker.newImuData.connect(self.on_new_imu_data)

        self.ahrs = self.create_ahrs()

        self.smm = SharedMemoryManager()
        for item in self.smi:
            if item[0] in [MOD_IMU, MOD_IMU_COUNT]:
                self.smm.create_variable(*item)

    def start_calibration(self):
        self.calibrating = True
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setText("Calibrating...")
        self.ahrs.restart()

        # Hold the current displayed position for 3 seconds
        self.calibration_timer.start(3000)


    def end_calibration(self):
        self.calibrating = False
        self.calibrate_button.setEnabled(True)
        self.calibrate_button.setText("Calibrate")

    def create_ahrs(self):
        ahrs = Ahrs()

        ahrs.set_settings(
            AhrsSettings(
                sample_rate=25,
                convention=CONVENTION_NWU,
                gain=0.5,
                gyroscope_range=2000,
                acceleration_rejection=10,
                magnetic_rejection=0,
                rejection_timeout=5 * 2000,
            )
        )

        return ahrs


    def on_new_imu_data(self, imu):
        roll, pitch, yaw = imu
        self.gizmo.setRotationMatrix(roll, pitch, yaw)

    def run(self):
        self.worker.start()
        self.resize(800, 600)
        ## Center on screen
        primaryScreen = QApplication.primaryScreen()
        assert primaryScreen is not None, "No primary screen found."

        screen = primaryScreen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(screen.center())
        self.move(frame.topLeft())

        ## Move to front
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.show()
        self.raise_()
        self.activateWindow()
        self.app.exec()

    def closeEvent(self, e):
        self.worker.stop()
        super().closeEvent(e)

class GestureWorker(QThread):
    newImuData = pyqtSignal(object)
    def __init__(self, gui):
        super().__init__()
        self.running=True
        self.gui=gui
        self.previousTime = time.perf_counter()

    def run(self):
        old_count = self.gui.smm.get_variable(MOD_IMU_COUNT)[0, 0]
        while self.running:
            new_count = self.gui.smm.get_variable(MOD_IMU_COUNT)[0, 0]
            if new_count == old_count:
                continue
            old_count = new_count

            imuData = self.gui.smm.get_variable(MOD_IMU)[0, :]
            roll, pitch, yaw = self.convertImuDataToAngles(imuData)
            
            self.newImuData.emit((roll, pitch, yaw))

    def convertImuDataToAngles(self, imuData):
        from scipy.spatial.transform import Rotation as R
        # Swap the two bytes of every int16 IMU value
        imuData = np.asarray(imuData, dtype=np.int16).byteswap()

        acceleration = imuData[0:3] / 1000.0  # Convert from mg to g
        gyro = imuData[3:6] 
        self.gui.ahrs.update_no_magnetometer(gyro, acceleration)

        q = self.gui.ahrs.get_quaternion()
        euler = R.from_quat(q).as_euler('xyz', degrees=False)

        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        return roll, pitch, yaw
         
    def stop(self):
        self.running=False
        self.wait()

class IMUGizmo(gl.GLViewWidget):
    def __init__(self):
        super().__init__()

        label_x = gl.GLTextItem(pos=(2.2, 0, 0), text='X', color=(255, 60, 60, 255))
        label_y = gl.GLTextItem(pos=(0, 2.2, 0), text='Y', color=(60, 255, 60, 255))
        label_z = gl.GLTextItem(pos=(0, 0, 2.2), text='Z', color=(60, 60, 255, 255))
        self.addItem(label_x)
        self.addItem(label_y)
        self.addItem(label_z) 

        self.setCameraPosition(distance=6, elevation=0, azimuth=180)
        a = gl.GLAxisItem(); a.setSize(2, 2, 2); self.addItem(a)


        base_path = Path(__file__).parent
        stl_path = base_path / "CAD files"/"EMaGer V3 CAD.stl"
        self.cylinder = load_stl(stl_path)

        self.cylinder.setColor((1.0, 0.4, 0.4, 1))
        self.cylinder.scale(0.01, 0.01, 0.01)
        self.cylinder.setGLOptions('opaque')

        self.addItem(self.cylinder)

        self.create_axes()


    def setRotationMatrix(self, roll, pitch, yaw):
        # update the transform of the cylinder based on roll, pitch, yaw 
        transform = QMatrix4x4()
        
        transform.rotate(180, 0, 0, 1)
        # I don't know why roll and yaw are inverted
        # Roll and yaw are negative because of the flipped axes (180 degrees rotation around z)
        # This is due to the EMaGer's IMU orientation
        transform.rotate(-roll * 180 / np.pi, 0, 0, 1)
        transform.rotate(pitch * 180 / np.pi, 0, 1, 0)
        transform.rotate(-yaw * 180 / np.pi, 1, 0, 0)
        self.cylinder.setTransform(transform)
        self.cylinder.scale(0.01, 0.01, 0.01)
        self.update_axes()

    def create_axes(self):
        # create axes attached to the cylinder
        self.axis_x = gl.GLLinePlotItem(
            pos=np.array([[0,0,0], [1,0,0]]),
            width=3,
            antialias=True
        )
        self.axis_y = gl.GLLinePlotItem(
            pos=np.array([[0,0,0], [0,1,0]]),
            width=3,
            antialias=True
        )
        self.axis_z = gl.GLLinePlotItem(
            pos=np.array([[0,0,0], [0,0,1]]),
            width=3,
            antialias=True
        )

        self.addItem(self.axis_x)
        self.addItem(self.axis_y)
        self.addItem(self.axis_z)

        # Labels for the body-frame axes (positions updated every frame)
        self.label_body_x = gl.GLTextItem(pos=(1, 0, 0), text='x', color=(255, 60, 60, 255))
        self.label_body_y = gl.GLTextItem(pos=(0, 1, 0), text='y', color=(60, 255, 60, 255))
        self.label_body_z = gl.GLTextItem(pos=(0, 0, 1), text='z', color=(60, 60, 255, 255))

        self.addItem(self.label_body_x)
        self.addItem(self.label_body_y)
        self.addItem(self.label_body_z)

    def update_axes(self):
        # update the axes based on the current transform of the cylinder
        T = self.cylinder.transform()

        R = np.array([
            [T.row(0).x(), T.row(0).y(), T.row(0).z()],
            [T.row(1).x(), T.row(1).y(), T.row(1).z()],
            [T.row(2).x(), T.row(2).y(), T.row(2).z()]
        ])

        origin = np.array([0, 0, 0])


        scale = 100
        self.axis_x.setData(pos=np.array([origin, R[:, 0] * scale]))
        self.axis_y.setData(pos=np.array([origin, R[:, 1] * scale]))
        self.axis_z.setData(pos=np.array([origin, R[:, 2] * scale]))

        # push labels out a bit past the tip so they don't overlap the lines
        self.label_body_x.setData(pos=R[:, 0] * scale)
        self.label_body_y.setData(pos=R[:, 1] * scale)
        self.label_body_z.setData(pos=R[:, 2] * scale)


def main(): 
    from emagerlib.utils.streamer_utils import get_emager_streamer

    process_emager, smi = get_emager_streamer()

    gui = GUI(smi) 
    gui.run()

    process_emager.terminate()


if __name__ == "__main__":
    main()