import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class CameraPerceptor(Node):
    def __init__(self):
        super().__init__('camera_perceptor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.get_logger().info('Camera Perceptor Node Started')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received image frame with size: {len(msg.data)} bytes')
        # Add perception logic here

def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
