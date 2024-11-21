import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from image_boundary import boundary_depth


class Orientation(Node):
    def __init__(self):
        super().__init__('orientation_publisher')
        self.publisher_vector = self.create_publisher(Float32MultiArray, '/gelsight/orientation', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # called every 100ms

        # Initialize GelSight and other related modules.
        self.nn, self.dev = boundary_depth(cam_id="GelSight Mini")

    def timer_callback(self):
        img = self.dev.get_raw_image()
        if img is None or img.size == 0:
            self.get_logger().error("Failed to capture image")
            return

        edges, img_small = self.depth_boundary_detection(img)
        vx, vy = self.extract_direction_vector(edges, img_small)

        if vx is not None and vy is not None:
            orientation_vector = Float32MultiArray()
            vx, vy = float(vx), float(vy)
            orientation_vector.data = [vx, vy]  # Use Float32MultiArray to publish
            self.publisher_vector.publish(orientation_vector)
        else:
            self.get_logger().warn("No valid direction vector detected")

    def depth_boundary_detection(self, img):
        img_small = cv2.resize(img, (320, 240))
        dm = self.nn.get_depthmap(img_small, mask_markers=True)

        ksize = 5
        sigma = 2.2
        gray_blur = cv2.GaussianBlur(dm, (ksize, ksize), sigma)

        gradient_x = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_magnitude_8U = cv2.convertScaleAbs(gradient_magnitude)
        texture = int(np.max(gradient_magnitude_8U))

        edges = cv2.Canny(gradient_magnitude_8U, texture, 2 * texture)
        return edges, img_small

    def extract_direction_vector(self, edges, img_small):
        # No edges
        points = np.column_stack(np.where(edges > 0))
        if points.shape[0] == 0:
            return None, None

        # orientation in 2D image coordinate system
        # Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=25, maxLineGap=10)

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_small, (x1, y1), (x2, y2), (0, 255, 128), 2)
                img_small[:, 1] = [255, 0, 0]  # red for x-axis
                img_small[1, :] = [0, 0, 255]  # blue for y-axis
                angle = np.arctan2(y2 - y1, x2 - x1)
                angles.append(angle)

            mean_angle = np.mean(angles)
            return np.cos(mean_angle), np.sin(mean_angle)

        # PCA method as a backup
        mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))
        return eigenvectors[1]


def main(args=None):
    rclpy.init(args=args)
    orientation_publisher = Orientation()
    rclpy.spin(orientation_publisher)
    orientation_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
