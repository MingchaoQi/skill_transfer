import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R, Slerp

class PoseControl(Node):
    def __init__(self):
        super().__init__('pose_control')

        # Publisher for target pose
        self.pose_pub = self.create_publisher(Pose, '/lbr/command/pose', 10)

        # Subscriber for current end effector pose
        self.pose_sub = self.create_subscription(
            Pose, '/lbr/state/pose', self.pose_callback, 10
        )

        # Subscriber for the 2D orientation data from Gelsight
        self.orientation_sub = self.create_subscription(
            Float32MultiArray, '/gelsight/orientation', self.control_callback, 10
        )

        self.current_pose = None
        self.max_step = 0.01
        self.max_angle_step = 0.1  # (in radians)

    def pose_callback(self, msg):
        # Store the received current pose
        self.current_pose = msg

    def control_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("Current pose is not available yet.")
            return

        direction_2d = np.array([msg.data[0], msg.data[1]])

        current_pose = self.current_pose

        desired_pose = self.simple_transform_and_align(direction_2d, current_pose)

        self.move_towards_pose(current_pose, desired_pose)

    def simple_transform_and_align(self, direction_2d, current_pose):
        # Map the 2D direction vector to an angle (angle with respect to the x-axis)
        # Calculate the angle with respect to the x-axis, returns range [-pi/2, pi/2]
        angle = np.arctan2(direction_2d[1], direction_2d[0])
        print(angle)

        rotation_quat = R.from_euler('y', angle, degrees=False).as_quat(False)  # Generate rotation around the y-axis

        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])

        new_quat = R.from_quat(current_quat) * R.from_quat(rotation_quat)

        desired_pose = Pose()
        desired_pose.position = current_pose.position

        new_quat_arr = new_quat.as_quat()
        desired_pose.orientation.x = new_quat_arr[0]
        desired_pose.orientation.y = new_quat_arr[1]
        desired_pose.orientation.z = new_quat_arr[2]
        desired_pose.orientation.w = new_quat_arr[3]

        return desired_pose

    def move_towards_pose(self, current_pose, desired_pose):
        # Interpolate position
        position_diff = np.array([
            desired_pose.position.x - current_pose.position.x,
            desired_pose.position.y - current_pose.position.y,
            desired_pose.position.z - current_pose.position.z
        ])

        distance = np.linalg.norm(position_diff)
        if distance > self.max_step:
            step = position_diff / distance * self.max_step
        else:
            step = position_diff

        # Interpolate quaternion (using spherical linear interpolation)
        current_quat = np.array([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])

        desired_quat = np.array([
            desired_pose.orientation.x,
            desired_pose.orientation.y,
            desired_pose.orientation.z,
            desired_pose.orientation.w
        ])

        # Calculate interpolation ratio
        angle_diff = np.arccos(np.clip(np.dot(current_quat, desired_quat), -1.0, 1.0)) * 2.0
        if angle_diff > self.max_angle_step:
            t = self.max_angle_step / angle_diff
        else:
            t = 1.0

        key_rots = R.from_quat([current_quat, desired_quat])
        slerp = Slerp([0, 1], key_rots)
        interpolated_quat = slerp(t).as_quat()

        intermediate_pose = Pose()
        intermediate_pose.position.x = current_pose.position.x + step[0]
        intermediate_pose.position.y = current_pose.position.y + step[1]
        intermediate_pose.position.z = current_pose.position.z + step[2]

        intermediate_pose.orientation.x = interpolated_quat[0]
        intermediate_pose.orientation.y = interpolated_quat[1]
        intermediate_pose.orientation.z = interpolated_quat[2]
        intermediate_pose.orientation.w = interpolated_quat[3]

        self.pose_pub.publish(intermediate_pose)

def main(args=None):
    rclpy.init(args=args)
    pose_control = PoseControl()
    rclpy.spin(pose_control)
    pose_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
