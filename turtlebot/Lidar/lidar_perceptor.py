import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarPerceptor(Node):
    def __init__(self):
        super().__init__('lidar_perceptor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.get_logger().info('Lidar Perceptor Node Started')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received scan with {len(msg.ranges)} ranges')
        # Add perception logic here

def main(args=None):
    rclpy.init(args=args)
    node = LidarPerceptor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
