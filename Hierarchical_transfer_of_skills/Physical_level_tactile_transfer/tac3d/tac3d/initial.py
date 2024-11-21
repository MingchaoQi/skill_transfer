import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from lbr_fri_idl.msg import LBRState
from lbr_fri_idl.msg import LBRJointPositionCommand
import numpy as np
import time

class LBRController(Node):
    def __init__(self):
        super().__init__('initial_controller')
        self.subscription = self.create_subscription(
            LBRState,
            '/lbr/state',
            self.current_joint_position_callback,
            10)
        self.publisher = self.create_publisher(
            LBRJointPositionCommand,
            '/lbr/command/joint_position',
            QoSProfile(depth=10))
        self.current_joint_position = None
        self.target_joint_position = [0.0, 0.349, 0.0, -1.222, 0.0, 1.571, 0.0]
        self.rate = 100

    def current_joint_position_callback(self, msg):
        self.current_joint_position = msg.measured_joint_position
        self.get_logger().info(f'Received positions: {self.current_joint_position}')
        self.send_joint_positions()

    def interpolate_joint_positions(self, start_positions, end_positions, duration):
        num_steps = int(duration * self.rate)
        trajectory = []

        for i in range(num_steps + 1):
            alpha = i / num_steps
            interpolated_positions = [
                start + alpha * (end - start)
                for start, end in zip(start_positions, end_positions)
            ]
            trajectory.append(interpolated_positions)

        return trajectory


    def send_joint_positions(self):
        if self.current_joint_position is not None:
            trajectory = self.interpolate_joint_positions(self.current_joint_position,
                                                          self.target_joint_position,
                                                          duration=10.0)
            for positions in trajectory:
                msg = LBRJointPositionCommand()
                msg.joint_position = positions
                self.publisher.publish(msg)
                self.get_logger().info(f'Published positions: {msg.joint_position}')
                time.sleep(1.0 / self.rate)
            self.get_logger().info('Finished sending joint positions. Shutting down...')
            self.destroy_node()
        else:
            self.get_logger().warn('Waiting for current joint positions...')

def main(args=None):
    rclpy.init(args=args)
    initial_controller = LBRController()
    rclpy.spin(initial_controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
