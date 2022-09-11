
import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread

from mina.sensor.vision.camera import Camera

class RealSense(Camera):
    def __init__(self, height=480, width=640, hardware_reset=False):
        # Hardware reset, equivalent to unplugging and plugging the camera again
        if hardware_reset:
            self.reset()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

        # self.align_to_color=rs.align(rs.stream.color)
        self.align = rs.align(rs.stream.color)
        # Start streaming

        # Records the intrinsic parameters
        cfg = self.pipeline.start(config)
        profile_color = cfg.get_stream(rs.stream.color) # Fetch stream profile for color stream
        profile_depth = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
        self._intrinsics_color = profile_color.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        self._intrinsics_depth = profile_depth.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    def get_intrinsics(self, sensor="color"):
        if sensor=="depth":
            i = self._intrinsics_depth
            return [i.fx, i.fy, i.ppx, i.ppy]
        i = self._intrinsics_color
        return [i.fx, i.fy, i.ppx, i.ppy]

    def reset(self):
        print("**** Resetting the RealSense device, might take second or two ****")
        import time
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
        time.sleep(2)
        print("**** Done Resetting! ****")

    def get_frames(self):
        try:
            frames_raw = self.pipeline.wait_for_frames()
            frames = self.align.process(frames_raw)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            if not depth_frame or not color_frame:
                return False, np.empty((0,0)), np.empty((0,0))
            return True, color_image, depth_image
        except RuntimeError as e:
            return False, np.empty((0,0)), np.empty((0,0))
    
    # def get_frames_generator(self):
    #     while True:
    #         yield self.get_frames()

    def run(self):
        self.is_running = True
        def callback():
            while self.is_running:
                try:
                    ret, color, depth = self.get_frames()
                    if ret:
                        self.color_image = color
                        self.depth_image = depth
                except Exception:
                    self.terminate()

        self.thread = Thread(target=callback).start()
    
    # def terminate(self):
    #     self.is_running = False
    
    def get_global_frame(self, align=False):
        frame = self.pipeline.wait_for_frames()
        if frame:
            if align:
                frame = rs.align(rs.stream.color).process(frame)
            return True, frame
        return False, frame


    # def visualize(self, frame):
    #     """Visualizes given data frame in a window"""
    #     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow('RealSense', frame)


    # def spin_once(self, duration, stop_key="q"):
    #     if cv2.waitKey(duration) & 0xFF == ord(stop_key):
    #         self.release()
    #         raise Exception("Stop interrupt received")

    # @staticmethod
    # def visualize(frame, frame_name="Realsense"):
    #     """Visualizes given data frame in a window"""
    #     cv2.namedWindow(frame_name, cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow(frame_name, frame)

    # @staticmethod
    # def spin_once(duration, stop_key="q"):
    #     if cv2.waitKey(duration) & 0xFF == ord(stop_key):
    #         # self.release()
    #         raise Exception("Stop interrupt received")

    def get_pointcloud(self):
        """Returns the pointcloud
        Raises:
            NotImplemented: since not implemented
        """
        raise NotImplemented


    def release(self):
        """stops the camera from driver level
        """
        self.pipeline.stop()

    
    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if __name__ == "__main__":
    # Using Threading to read images
    cam = RealSense(height=480, width=640)
    while True:
        _,color,depth = cam.get_frames()
        cam.visualize(color)
        cam.spin_once(10)
    
    # Using generator to read images
    cam = RealSense(height=480, width=640)
    stream = cam.get_frames_generator()
    while True:
        _,color,depth = stream.__next__()
        cam.visualize(color)
        cam.spin_once(10)
