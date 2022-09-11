
import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread

class Camera:
    def __init__(self, height=480, width=640, source=0):
        # Configure color streams
        print(f"Using the index {source}")
        self.video = cv2.VideoCapture(source)

    def get_frames(self):
        """Return the recent frame capture by the sensor

        Returns:
            tuple: (bool: result, cv2-image: image)
        """
        return self.video.read()
    
    def get_frames_generator(self):
        """Return a generator object which provides recent sensor frame, call __next__() to get recent frames

        Yields:
            frame: the recent frame buffer in the sensor memory
        """
        while True:
            yield self.get_frames()

    def run(self):
        """Runs the video in thread, data is kept in color_image
        """
        self.is_running = True
        def callback():
            while self.is_running:
                try:
                    ret, color = self.get_frames()
                    if ret:
                        self.color_image = color
                except Exception:
                    self.terminate()

        self.thread = Thread(target=callback).start()
    

    def terminate(self):
        """ terminate the thread running and releases the sensor lock by the class
        """
        self.is_running = False
        self.release()


    def visualize(self, frame):
        """Visualizes given data frame in a window"""
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', frame)


    def spin_once(self, duration, stop_key="q"):
        """Handles keyboard operation and presents delay between frames, this is object method

        Args:
            duration (float): delay between frames in milliseconds
            stop_key (str, optional): the key used to stop the visualization. Defaults to "q".

        Raises:
            Exception: raised when the stop key in pressed, indicates stopping sensor and releasing lock
        """
        if cv2.waitKey(duration) & 0xFF == ord(stop_key):
            self.release()
            raise Exception("Stop interrupt received")

    @staticmethod
    def visualize(frame, frame_name="Realsense"):
        """Visualizes given data frame in a window

        Args:
            frame (ndarray, CVMAT): opencv compatible image
            frame_name (str, optional): title of visulaizatin window. Defaults to "Realsense".
        """
        cv2.namedWindow(frame_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(frame_name, frame)

    @staticmethod
    def spin_once(duration, stop_key="q"):
        """Handles keyboard operation and presents delay between frames, remember this is classmethod and cant be used with class objects

        Args:
            duration (float): delay between frames in milliseconds
            stop_key (str, optional): the key used to stop the visualization. Defaults to "q".

        Raises:
            Exception: raised when the stop key in pressed, indicates stopping sensor and releasing lock
        """
        if cv2.waitKey(duration) & 0xFF == ord(stop_key):
            # self.release()
            raise Exception("Stop interrupt received")

    def release(self):
        """Stops the camera from driver level
        """
        self.video.release()


if __name__ == "__main__":
    # Using Threading to read images
    cam = Camera(height=480, width=640, source=4) # source 4 for realsense camera
    while True:
        _,color = cam.get_frames()
        cam.visualize(color)
        cam.spin_once(10)
    
    # Using generator to read images
    cam = RealSense(height=480, width=640)
    stream = cam.get_frames_generator()
    while True:
        _,color,depth = stream.__next__()
        cam.visualize(color)
        cam.spin_once(10)
