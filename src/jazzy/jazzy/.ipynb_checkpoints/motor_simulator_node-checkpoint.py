
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from collections import deque
import can
import struct
import socket 


class MotorSimulatorNode(Node):
    def __init__(self):
        super().__init__('motor_simulator_node')
        self.subscription = self.create_subscription(
            Point,
            '/nearest_person',
            self.listener_callback,
            10
        )
        self.bus = None
        try:
            self.bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=500000) 
            self.get_logger().info("CAN bus initialized successfully.")
        except Exception as e:
            self.get_logger().info(f"Failed to initialize CAN bus: {e}")
            self.get_logger().info("CAN messages will not be sent.")
            
    def can_send_pi(self, speed, angle, distance_to_target): 
        speed = int(round(speed))
        angle = int(round(angle))
    
        self.get_logger().info(f'CAN Send -> Speed In: {speed}, Steer Adjust In: {angle}, Dist: {distance_to_target:.2f}m')
        self.get_logger().info(f'           -> CAN Values: Speed={speed}, Steer={angle}')
        
        if self.bus is not None: 
            try:
                # Nếu thiết bị cần dữ liệu 16-bit big-endian, chỉ dùng struct.pack thôi
                data = struct.pack('>hh', speed, angle)
                
                msg = can.Message(arbitration_id=0x21, data=data, is_extended_id=False)
                self.bus.send(msg)
    
            except Exception as e:
                self.get_logger().info(f"Error sending CAN message: {e}")
        else:
            self.get_logger().info("CAN bus not available. Message not sent.")
        

    def listener_callback(self, msg: Point):
        x = msg.x
        y = msg.y
        z = msg.z
        angle = -(x-160)/160*90 + 90 
        speed = -25
        if z > 0.3:
            speed = 25
        self.can_send_pi(speed, angle, z)

        
def main(args=None):
    rclpy.init(args=args)
    node = MotorSimulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
