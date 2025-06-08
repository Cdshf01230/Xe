import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
import threading
import traceback
import time


class Camera:
    def __init__(self, 
                 fps=30, 
                 width=640, 
                 height=480, 
                 openni_libs='libs/'):
        self.fps = fps
        self.width = width
        self.height = height
        self.openni_libs = openni_libs
        self.wait_time = int(1000.0/float(fps))
        self.load()
    
    def unload(self):
        openni2.unload()
        
    def load(self):
        try:
            openni2.initialize(self.openni_libs)
            self.dev = openni2.Device.open_any()
            
            self.depth_stream = self.dev.create_depth_stream()
            self.depth_stream.set_video_mode(
                c_api.OniVideoMode(
                    pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, 
                    resolutionX = self.width, 
                    resolutionY = self.height, 
                    fps = self.fps
                )
            )
            self.depth_stream.start()
            
            self.color_stream = self.dev.create_color_stream()
            self.color_stream.set_video_mode(
                c_api.OniVideoMode(
                    pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, 
                    resolutionX = self.width, 
                    resolutionY = self.height, 
                    fps = self.fps
                )
            )
            self.color_stream.start()
            
            self.dev.set_image_registration_mode(
                c_api.OniImageRegistrationMode.ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR
            )
            print("OpenNI camera loaded successfully")
        except:
            print('Cannot load OpenNI camera')
            traceback.print_exc()
        
    def get_depth(self):
        frame = self.depth_stream.read_frame()
        if frame is None:
            return None
        frame_data = frame.get_buffer_as_uint16()
        img = np.frombuffer(frame_data, dtype=np.uint16)
        try:
            img = img.reshape((self.height, self.width))
        except Exception as e:
            print(f"Depth image reshape error: {e}")
            return None
        return img
        
    def get_color(self):
        frame = self.color_stream.read_frame()
        if frame is None:
            return None
        frame_data = frame.get_buffer_as_uint8()
        colorPix = np.frombuffer(frame_data, dtype=np.uint8)
        try:
            colorPix = colorPix.reshape((self.height, self.width, 3))
        except Exception as e:
            print(f"Color image reshape error: {e}")
            return None
        return colorPix
    
    def get_depth_and_color(self):
        return self.get_depth(), self.get_color()


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera')
        path = os.path.join(
            get_package_share_directory('jazzy'),
            'libs'
        )
        
        self.cam = Camera(openni_libs=path)
        self.bridge = CvBridge()

        self.pub_color = self.create_publisher(Image, '/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/depth/image_raw', 10)

        self.latest_rgb = None
        self.latest_depth = None
        self.lock = threading.Lock()

        self.read_thread = threading.Thread(target=self.read_frames_loop, daemon=True)
        self.read_thread.start()

        self.timer = self.create_timer(1/self.cam.fps, self.timer_callback)
        self.get_logger().info("Camera node started")

        self.last_publish_time = None  # Thời gian lần publish trước đó

    def read_frames_loop(self):
        while rclpy.ok():
            try:
                rgb = self.cam.get_color()
                depth = self.cam.get_depth()
                if rgb is not None and depth is not None:
                    with self.lock:
                        self.latest_rgb = rgb
                        self.latest_depth = depth
                else:
                    self.get_logger().warn('Không nhận được frame RGB hoặc Depth')
            except Exception as e:
                self.get_logger().error(f'Lỗi khi đọc frame camera: {e}')
            time.sleep(1/self.cam.fps)

    def timer_callback(self):
        with self.lock:
            rgb = self.latest_rgb
            depth = self.latest_depth

        if rgb is None or depth is None:
            self.get_logger().warn('Chưa có ảnh RGB hoặc Depth để publish.')
            return

        now = time.time()
        if self.last_publish_time is None:
            fps = 0.0
        else:
            dt = now - self.last_publish_time
            fps = 1.0 / dt if dt > 0 else 0.0
        self.last_publish_time = now

        timestamp = self.get_clock().now().to_msg()

        try:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            msg_color = self.bridge.cv2_to_imgmsg(rgb_bgr, encoding="bgr8")
            msg_color.header.stamp = timestamp
            msg_color.header.frame_id = "camera_color_frame"

            msg_depth = self.bridge.cv2_to_imgmsg(depth, encoding="passthrough")
            msg_depth.header.stamp = timestamp
            msg_depth.header.frame_id = "camera_depth_frame"

            self.pub_color.publish(msg_color)
            self.pub_depth.publish(msg_depth)

            self.get_logger().info(f'Publishing images at {fps:.2f} FPS')

        except Exception as e:
            self.get_logger().error(f'Lỗi khi convert hoặc publish ảnh: {e}')



def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
