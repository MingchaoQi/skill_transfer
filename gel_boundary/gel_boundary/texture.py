import cv2
import numpy as np
from gelsight import gsdevice
from image_boundary import boundary_depth
import time
from image_flow import Flow


def zoom(real_calibre, edges):
    """
    Calculate the image zoom ratio, comparing the maximum pixel distance with the real diameter of a standard circular aperture.
    :param real_calibre: Known diameter of the standard circular aperture.
    :param edges: Edge-detected image matrix reflecting the size and location of edges (non-zero at edges).
    :return: Image zoom ratio as pixel distance (in pixels) / real distance (in mm).
    """
    edge_coordinates = []
    height, width = edges.shape
    print('Aspect ratio of the image:', height, ':', width)
    for y in range(height):
        for x in range(width):
            if edges[y, x] != 0:
                edge_coordinates.append((x, y))

    max_x_diff = max_y_diff = 0
    for coord1 in edge_coordinates:
        for coord2 in edge_coordinates:
            x_diff, y_diff = abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1])
            max_x_diff, max_y_diff = max(max_x_diff, x_diff), max(max_y_diff, y_diff)

    zoom_ratio = (max_x_diff + max_y_diff) / real_calibre
    print('Image zoom ratio:', zoom_ratio)
    return zoom_ratio


def depth_boundary_detection(img, flag):
    """
    Detect depth boundaries in the image and process using filters and gradient calculations.
    :param img: Input image.
    :param flag: Boolean flag to activate certain processes after set intervals.
    :return: List of contour points detected in the image.
    """
    img_raw = cv2.resize(img, (320, 240))
    img_small = img_raw.copy()
    dm = nn.get_depthmap(img_small, mask_markers=True)

    contour_points = []
    if flag:
        global stable_region
        stable_region = find_stable_region(dm)

    ksize, sigma = 5, 2.2
    gray_blur = cv2.GaussianBlur(dm, (ksize, ksize), sigma)
    gradient_x = cv2.Sobel(dm, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(dm, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude_8U = cv2.convertScaleAbs(gradient_magnitude)

    texture = int(np.max(gradient_magnitude_8U))
    print('max:', np.max(gradient_magnitude_8U), 'texture:', texture)
    edges = cv2.Canny(gradient_magnitude_8U, texture, 2 * texture)

    img_small[edges != 0] = [255, 0, 0]
    img_edge = img_small.copy()

    points = np.column_stack(np.where(edges > 0))
    if points.shape[0]:
        mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=np.array([]))
        vx, vy, x0, y0 = eigenvectors[1], mean[0]
        x1, y1, x2, y2 = int(x0 + vx * 100), int(y0 + vy * 100), int(x0 - vx * 100), int(y0 - vy * 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=25, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_small, (x1, y1), (x2, y2), (0, 255, 128), 2)

    edges_dm = cv2.threshold(gradient_magnitude, 2, 255, cv2.THRESH_BINARY)[1]

    window_width, window_height = 640, 480
    resized_edges_dm = cv2.resize(edges_dm, (window_width, window_height))
    resized_gradient_magnitude = cv2.resize(gradient_magnitude, (window_width, window_height))
    resized_img_small = cv2.resize(img_small, (window_width, window_height))

    cv2.imshow('Edges', resized_edges_dm)
    cv2.imshow('Gradient Magnitude', resized_gradient_magnitude)
    cv2.imshow('Contours Image', resized_img_small)

    if flag:
        timestamp, file_name = str(time.time()).replace(".", ""), f"stable_data/image_{timestamp}"
        gradient_magnitude_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(file_name + '.jpg', img_raw)
        cv2.imwrite(file_name + 'w.jpg', img_small)
        cv2.imwrite(file_name + 'g.jpg', gradient_magnitude_normalized)
        cv2.imwrite(file_name + 'd.jpg', img_edge)

    return contour_points


previous_depth_image, interval, flag, stable_region = None, 0.5, 0, np.zeros([240, 320], dtype=np.uint8)

nn, dev = boundary_depth(cam_id="GelSight Mini")

while True and cv2.waitKey(1) == -1:
    img = dev.get_raw_image()
    if img is None or img.size == 0:
        print("Error: Unable to get image from camera.")
        continue

    contour_points_depth = depth_boundary_detection(img, flag)
    flag = 1

cv2.destroyAllWindows()

