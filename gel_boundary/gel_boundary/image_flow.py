#!/usr/local/bin/python
# encoding: utf-8
import cv2
import numpy as np
import numpy.matlib
import time
from gelsight import gsdevice
import matplotlib.pyplot as plt

class Flow:
    """
    A class to handle optical flow operations and video output for visualizing flow in an optical sensing application.
    """
    def __init__(self, col, row):
        self.x0 = np.matlib.repmat(np.arange(row), col, 1).T
        self.y0 = np.matlib.repmat(np.arange(col), row, 1)
        self.x = np.zeros_like(self.x0, dtype=int)
        self.y = np.zeros_like(self.y0, dtype=int)
        self.col = col
        self.row = row

        self.out = cv2.VideoWriter('flow picture/flow.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5, (col, row))
        self.out_2 = cv2.VideoWriter('flow_init picture/flow_init.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5, (col, row))
        self.out_3 = cv2.VideoWriter('flow picture/flow_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5, (col, row))
        self.out_4 = cv2.VideoWriter('flow_init picture/flow_init_2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3.5, (col, row))

    def get_raw_img(self, frame):
        """
        Resize and crop the frame to fit the sensor dimensions and remove unwanted borders.
        """
        img = cv2.resize(frame, (895, 672))
        border_size = int(img.shape[0] * (1 / 7))
        img = img[border_size:-border_size, border_size:-border_size]
        img = img[:, :-1]
        return cv2.resize(img, (self.col, self.row))

    def add_flow(self, flow):
        """
        Update the flow fields by adding the calculated flow to the existing flow matrix.
        """
        dx = np.round(self.x + self.x0).astype(int).clip(0, self.row - 1)
        dy = np.round(self.y + self.y0).astype(int).clip(0, self.col - 1)
        ds = np.reshape(flow[np.reshape(dx, -1), np.reshape(dy, -1)], (self.row, self.col, -1))
        self.x += ds[:, :, 0]
        self.y += ds[:, :, 1]
        return self.x, self.y

    def flow2color(self, flow, hsv, K=15):
        """
        Convert flow vectors into an RGB color map to visualize the magnitude and direction of the flow.
        """
        mag, ang = cv2.cartToPolar(-flow[..., 1], flow[..., 0])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = np.clip(mag * K * 960 / self.col, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def draw(self, img, flow, scale=2.0):
        """
        Draw flow vectors on the image using arrowed lines to represent the optical flow direction and magnitude.
        """
        step = 10
        for i in range(10, self.row, step):
            for j in range(10, self.col, step):
                d = (flow[i, j] * scale).astype(int)
                cv2.arrowedLine(img, (j, i), (j + d[0], i + d[1]), (0, 255, 255), 1)
        return img

    def cal_force(self, flow, delta):
        """
        Calculate average forces and torque based on the flow vectors.
        """
        force_x = np.mean(flow[:, :, 0])
        force_y = np.mean(flow[:, :, 1])
        torque = np.mean(np.gradient(flow[:, :, 1])[1] - np.gradient(flow[:, :, 0])[0]) / delta
        return force_x, force_y, torque

    def draw_sumLine(self, img, center, sum_x, sum_y, scale=5.0):
        """
        Draw a summary line representing the aggregate vector at the center of the object.
        """
        end = (int(center[0] + sum_x * scale), int(center[1] + sum_y * scale))
        cv2.arrowedLine(img, (int(center[0]), int(center[1])), end, (0, 0, 255), 2)

    def heatmap(self, flow, min_flow, max_flow):
        """
        Generate a heatmap based on the magnitude of flow vectors.
        """
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitude = np.clip(magnitude, min_flow, max_flow)
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    def cal_div_curl(self, v_field, delta):
        """
        Calculate divergence and curl of the vector field to analyze the fluid dynamics of the flow.
        """
        div = np.mean(np.gradient(v_field[:, :, 0])[1] + np.gradient(v_field[:, :, 1])[0]) / delta
        curl = np.mean(np.gradient(v_field[:, :, 1])[1] - np.gradient(v_field[:, :, 0])[0]) / delta
        return div, curl

    def cal_gripper_force(self, flow_left, flow_right, delta):
        """
        Calculate forces on the gripper based on the flow from two different sensors.
        """
        force_x_left, force_y_left, torque_left = self.cal_force(flow_left, delta)
        div_left, curl_left = self.cal_div_curl(flow_left, delta)
        force_x_right, force_y_right, torque_right = self.cal_force(flow_right, delta)
        div_right, curl_right = self.cal_div_curl(flow_right, delta)
        F_x = -(force_x_left + force_x_right)
        F_y = (curl_left - curl_right) / 2
        F_z = (force_x_left - force_x_right) / 2
        T_z = (force_y_left - force_y_right)
        return F_x, F_y, F_z, T_z

    def cal_flow(self, f0, f0_2, frame2_1, frame2_2):
        """
        Calculate the optical flow between initial frames and current frames for two sensors.
        """
        next_1 = cv2.cvtColor(frame2_1, cv2.COLOR_BGR2GRAY)
        next_2 = cv2.cvtColor(frame2_2, cv2.COLOR_BGR2GRAY)
        flow_left = cv2.calcOpticalFlowFarneback(f0, next_1, None, 0.5, 3, int(180 * self.col / 960), 5, 5, 1.2, 0)
        flow_right = cv2.calcOpticalFlowFarneback(f0_2, next_2, None, 0.5, 3, int(180 * self.col / 960), 5, 5, 1.2, 0)
        frame3_1 = self.draw(np.copy(frame2_1), flow_left)
        frame3_2 = self.draw(np.copy(frame2_2), flow_right)
        self.out.write(frame3_1)
        self.out_2.write(frame2_1)
        self.out_3.write(frame3_2)
        self.out_4.write(frame2_2)
        cv2.imshow('frame', frame3_1)
        cv2.imshow('frame2', frame3_2)
        return flow_left, flow_right, frame3_1, frame3_2, next_1, next_2

if __name__ == "__main__":
    # Sensor initialization
    cap = cv2.VideoCapture(0)
    cap_2 = cv2.VideoCapture(2)
    Flow = Flow(col=320, row=240)
    _, frame1 = cap.read()
    frame1 = Flow.get_raw_img(frame1)
    f0 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    _, frame1_2 = cap_2.read()
    frame1_2 = Flow.get_raw_img(frame1_2)
    f0_2 = cv2.cvtColor(frame1_2, cv2.COLOR_BGR2GRAY)

    while True:
        _, frame2_1 = cap.read()
        frame2_1 = Flow.get_raw_img(frame2_1)
        next_1 = cv2.cvtColor(frame2_1, cv2.COLOR_BGR2GRAY)
        _, frame2_2 = cap_2.read()
        frame2_2 = Flow.get_raw_img(frame2_2)
        next_2 = cv2.cvtColor(frame2_2, cv2.COLOR_BGR2GRAY)
        flow_left, flow_right, _, _, _, _ = Flow.cal_flow(f0, f0_2, frame2_1, frame2_2)
        div_left, curl_left = Flow.cal_div_curl(flow_left, delta=1)
        force_x

