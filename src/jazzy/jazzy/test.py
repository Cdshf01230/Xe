import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
import threading
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO 
import time


class YoloDetector:
    def __init__(self, conf_thres=0.3):
        model_path = os.path.join(
            get_package_share_directory('jazzy'),
            'model',
            'best.pt'  # Model YOLO đã được huấn luyện nhận diện người
        )
        print("Model path:", model_path)
        assert os.path.exists(model_path), "Model .pt không tồn tại!"

        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def predict(self, image):
        results = self.model.predict(
            source=image,
            conf=self.conf_thres,
            classes=[0],  # class 0 là 'person' trong COCO
            save=False,
            imgsz=320,
            verbose=False,
            device=0  # set -1 nếu không có GPU
        )

        boxes = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf.cpu().numpy())
                w = int(x2 - x1)
                h = int(y2 - y1)
                boxes.append({
                    'box': (int(x1), int(y1), w, h),
                    'conf': conf
                })
        return boxes


class NearestPersonNode(Node):
    def __init__(self):
        super().__init__('nearest_person_node')
        self.bridge = CvBridge()
        self.yolo = YoloDetector(conf_thres=0.3)

        self.color_sub = Subscriber(self, Image, '/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/depth/image_raw')
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.cache_callback)

        self.pub = self.create_publisher(Point, '/nearest_person', 10)

        self.latest_color = None
        self.latest_depth = None
        self.lock = threading.Lock()
        self.processing = False

        self.frame_count = 0
        self.start_time = time.time()

        self.timer = self.create_timer(0.1, self.process_latest_frame)

        self.get_logger().info('Nearest person node started.')

    def cache_callback(self, color_msg, depth_msg):
        with self.lock:
            self.latest_color = color_msg
            self.latest_depth = depth_msg

    def process_latest_frame(self):
        if self.processing:
            return

        with self.lock:
            if self.latest_color is None or self.latest_depth is None:
                return
            color_msg = self.latest_color
            depth_msg = self.latest_depth
            self.latest_color = None
            self.latest_depth = None

        self.processing = True
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            results = self.yolo.predict(color_img)

            nearest = None
            min_dist = float('inf')

            for result in results:
                x, y, w, h = result['box']
                cx = x + w // 2
                cy = y + h // 2

                if not (0 <= cx < depth_img.shape[1] and 0 <= cy < depth_img.shape[0]):
                    continue

                region = depth_img[max(0, cy-1):cy+2, max(0, cx-1):cx+2]
                valid = region[region > 0]

                if valid.size == 0:
                    continue

                avg_depth = np.mean(valid)
                if depth_msg.encoding == '16UC1':
                    distance = avg_depth / 1000.0
                elif depth_msg.encoding == '32FC1':
                    distance = avg_depth
                else:
                    continue

                if distance < min_dist:
                    min_dist = distance
                    nearest = (cx, cy)

            point = Point()
            if nearest and min_dist != float('inf'):
                point.x = float(nearest[0])
                point.y = float(nearest[1])
                point.z = float(min_dist)
                self.get_logger().info(f'Nearest person at ({point.x}, {point.y}) = {point.z:.2f}m')
            else:
                point.x = 0.0
                point.y = 0.0
                point.z = 0.0
                self.get_logger().info('No valid person detected.')

            self.pub.publish(point)

            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                fps = self.frame_count / elapsed
                self.get_logger().info(f"Processing FPS: {fps:.2f}")
                self.frame_count = 0
                self.start_time = time.time()

        except Exception as e:
            self.get_logger().error(f'Error processing frame: {e}')
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = NearestPersonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
