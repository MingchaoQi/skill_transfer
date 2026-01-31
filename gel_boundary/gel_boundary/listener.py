# listener_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateListener(Node):

    def __init__(self):
        super().__init__('joint_state_listener')
        self.position = None
        self.subscription = self.create_subscription(
            JointState,
            '/lbr/joint_states',
            self.joint_state_callback,
            10)
        self.subscription  # prevent unused variable warning

    def joint_state_callback(self, msg):
        self.get_logger().info(f'Received joint states:\n{msg}')
        self.position = msg.position
        self.get_logger().info(f'Received positions: \n{self.position}')

    def run(self):
        while True:
            if self.position is not None:
                self.get_logger().info(f'Received positions: {self.position}')
            else:
                self.get_logger().warn('Waiting for current joint positions...')

def main(args=None):
    rclpy.init(args=args)
    node = JointStateListener()
    rclpy.spin(node)
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

