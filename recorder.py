#!python3
#from ctypes import *
import os
import pyrealsense2 as rs
import numpy as np
import open3d
import cv2
import argparse
import os, datetime, sys

parser = argparse.ArgumentParser(
    prog='rstest',
    description='utility program for realsense'
    )

parser.add_argument('-a', '--aligned', default=True, help='is aligned')
parser.add_argument('-s', '--size', default='1280x720', help='image size (wxh)')
parser.add_argument('-o', '--outdir', default='dump', help='directory path of recorded data')
parser.add_argument('-f', '--fps', default=15, type=int, help='fps')

args = parser.parse_args()
w,h = args.size.split('x')
w = int(w)
h = int(h)

pipeline = rs.pipeline()
conf = rs.config()
conf.enable_stream(rs.stream.color, w, h, rs.format.bgr8, args.fps)
conf.enable_stream(rs.stream.depth, w, h, rs.format.z16, args.fps)
prof = pipeline.start(conf)
depth_sensor = prof.get_device().first_depth_sensor()


rsintrinsics = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intrinsics = open3d.camera.PinholeCameraIntrinsic(rsintrinsics.width, rsintrinsics.height, rsintrinsics.fx, rsintrinsics.fy, rsintrinsics.ppx, rsintrinsics.ppy)

def create_point_cloud(colorimg, depthimg, intrinsics):
    depth = open3d.geometry.Image(np.asanyarray(depthimg))
    color = open3d.geometry.Image(colorimg)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    #pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    return pcd

clipping_distance_in_meters = 2 #1 meter
clipping_distance = clipping_distance_in_meters / depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

COLOR_MODE = 0
DEPTH_MODE = 1
POINT_MODE = 2
mode = COLOR_MODE


try:
    while True:

        frames = pipeline.wait_for_frames()

        if args.aligned:
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.first(rs.stream.color)
            depth_frame = aligned_frames.get_depth_frame()
        else:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if mode == POINT_MODE:
            pcd = create_point_cloud(color_image, depth_image, intrinsics)
            open3d.visualization.draw_geometries([pcd])
        else:
            if  mode == DEPTH_MODE:
                image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            else:
                image = color_image.copy()
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', image)

                c = cv2.waitKey(1)

        if c == 27 or c == ord('q'):
            break
        elif c == ord('s'):
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            #cv2.imwrite("{}/color{:06d}.png".format(outdir, cnt), color_image)
            #cv2.imwrite("{}/depth{:06d}.png".format(outdir, cnt), depth_image)
            pcd = create_point_cloud(color_image, depth_image, intrinsics)
            open3d.write_point_cloud('{}/pc{:06d}.pcd'.format(doutdir, cnt), pcd)
            cnt = cnt + 1
        elif c == ord('t'):
            mode = (mode + 1) % 3


finally:
    pipeline.stop()



