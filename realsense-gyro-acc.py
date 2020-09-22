import pyrealsense2 as rs
import numpy as np
import time


W = 640
H = 480
FPS = 30

def initialize_camera():
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel)
    conf.enable_stream(rs.stream.gyro)

    prof = p.start(conf)
    return p

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

p = initialize_camera()
t0 = time.time()
try:
    while True:
        t1 = time.time()
        fs = p.wait_for_frames()
        accel = accel_data(fs[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(fs[1].as_motion_frame().get_motion_data())
        print(int((t1-t0) * 1000), accel, gyro)
        t0 = t1

finally:
    p.stop()