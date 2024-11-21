import numpy as np
import math
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
import robosuite as suite
from robosuite_task_zoo.environments.manipulation import HammerPlaceEnv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splrep, splev


class Motion_planning:
    """
    Class for motion planning, including obstacle mapping, path searching, and path smoothing.
    """

    def __init__(self, env, dx, dy, dz, gripper_open=True):
        """
        Initialize the motion planning instance.

        Args:
            env: The simulation environment.
            dx (float): Resolution in the x direction.
            dy (float): Resolution in the y direction.
            dz (float): Resolution in the z direction.
            gripper_open (bool): Whether the gripper is open.
        """
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.env = env
        self.gripper_open = gripper_open

    def range_box_G(
        self,
        start,
        target,
        start_range=np.array([0.02, 0.02, 0.02, 0.01]),
        target_range=np.array([0.03, 0.03, 0.02]),
    ):
        """
        Define a voxelized map for motion planning, considering gripper state and obstacle avoidance.

        Args:
            start (ndarray): Starting position.
            target (ndarray): Target position.
            start_range (ndarray): Voxel range around the starting point.
            target_range (ndarray): Voxel range around the target point.
        """
        point1 = start
        obstacle1 = self.env.door_pos
        obstacle2 = self.env._cabinet_base_center_xpos
        point2 = target
        obstacle1_box = np.array([0.025, 0.03, 0.06])
        obstacle2_box = np.array([0.013, 0.015, 0.015])

        gripper_mini = np.array(
            [
                point1[0] - start_range[0],
                point1[1] - start_range[1],
                point1[2] - start_range[3],
            ]
        )
        gripper_max = np.array(
            [
                point1[0] + start_range[0],
                point1[1] + start_range[1],
                point1[2] + start_range[2],
            ]
        )

        Points = np.array(
            [
                point1,
                point2,
                obstacle1,
                obstacle2,
                obstacle1 - obstacle1_box / 2,
                obstacle1 + obstacle1_box / 2,
                obstacle2 - obstacle2_box / 2,
                obstacle2 + obstacle2_box / 2,
                point2 - target_range / 2,
                point2 + target_range / 2,
                gripper_mini,
                gripper_max,
            ]
        ).reshape(-1, 3)
        self.min_x, self.max_x = np.min(Points[:, 0]), np.max(Points[:, 0])
        self.min_y, self.max_y = np.min(Points[:, 1]), np.max(Points[:, 1])
        self.min_z, self.max_z = np.min(Points[:, 2]), np.max(Points[:, 2])

        self.height = int((self.max_z - self.min_z) / self.dz) + 1
        self.length = int((self.max_x - self.min_x) / self.dx) + 1
        self.width = int((self.max_y - self.min_y) / self.dy) + 1
        self.Map = np.zeros(shape=(self.length, self.width, self.height))
        self.gripper_range = [
            int(start_range[0] / self.dx),
            int(start_range[1] / self.dy),
            int(start_range[2] / self.dz),
            int(start_range[3] / self.dz),
        ]
        self.target_range = [
            int(target_range[0] / self.dx),
            int(target_range[1] / self.dy),
            int(target_range[2] / self.dz),
        ]

        self.start_point = [
            int((point1[0] - self.min_x) / self.dx),
            int((point1[1] - self.min_y) / self.dy),
            int((point1[2] - self.min_z) / self.dz),
        ]
        self.end_point = [
            int((point2[0] - self.min_x) / self.dx),
            int((point2[1] - self.min_y) / self.dy),
            int((point2[2] - self.min_z) / self.dz),
        ]

        obstacle1_point = [
            int((obstacle1[0] - self.min_x) / self.dx),
            int((obstacle1[1] - self.min_y) / self.dy),
            int((obstacle1[2] - self.min_z) / self.dz),
        ]
        obstacle2_point = [
            int((obstacle2[0] - self.min_x) / self.dx),
            int((obstacle2[1] - self.min_y) / self.dy),
            int((obstacle2[2] - self.min_z) / self.dz),
        ]

        for i in range(
            max(int(obstacle1_point[0] - obstacle1_box[0] / (2 * self.dx)), 0),
            min(
                int(obstacle1_point[0] + obstacle1_box[0] / (2 * self.dx)), self.length
            ),
            1,
        ):
            for j in range(
                max(int(obstacle1_point[1] - obstacle1_box[1] / (2 * self.dy)), 0),
                min(
                    int(obstacle1_point[1] + obstacle1_box[1] / (2 * self.dy)),
                    self.width,
                ),
                1,
            ):
                for k in range(
                    max(int(obstacle1_point[2] - obstacle1_box[2] / (2 * self.dz)), 0),
                    min(
                        int(obstacle1_point[2] + obstacle1_box[2] / (2 * self.dz)),
                        self.height,
                    ),
                    1,
                ):
                    self.Map[i][j][k] = 1
        for i in range(
            max(int(obstacle2_point[0] - obstacle2_box[0] / (2 * self.dx)), 0),
            min(
                int(obstacle2_point[0] + obstacle2_box[0] / (2 * self.dx)), self.length
            ),
            1,
        ):
            for j in range(
                max(int(obstacle2_point[1] - obstacle2_box[1] / (2 * self.dy)), 0),
                min(
                    int(obstacle2_point[1] + obstacle2_box[1] / (2 * self.dy)),
                    self.width,
                ),
                1,
            ):
                for k in range(
                    max(int(obstacle2_point[2] - obstacle2_box[2] / (2 * self.dz)), 0),
                    min(
                        int(obstacle2_point[2] + obstacle2_box[2] / (2 * self.dz)),
                        self.height,
                    ),
                    1,
                ):
                    self.Map[i][j][k] = 1

        if self.gripper_open:

            for i in range(
                max(int(self.end_point[0] - target_range[0] / (2 * self.dx)), 0),
                min(
                    int(self.end_point[0] + target_range[0] / (2 * self.dx)),
                    self.length,
                ),
                1,
            ):
                for j in range(
                    max(int(self.end_point[1] - target_range[1] / (2 * self.dy)), 0),
                    min(
                        int(self.end_point[1] + target_range[1] / (2 * self.dy)),
                        self.width,
                    ),
                    1,
                ):
                    for k in range(
                        max(
                            int(self.end_point[2] - target_range[2] / (2 * self.dz)), 0
                        ),
                        min(
                            int(self.end_point[2] + target_range[2] / (2 * self.dz)),
                            self.height,
                        ),
                        1,
                    ):
                        self.Map[i][j][k] = 1

            self.end_point = [
                int((point2[0] - self.min_x) / self.dx),
                int((point2[1] - self.min_y) / self.dy),
                int(
                    (point2[2] + target_range[2] + start_range[3] + 0.01 - self.min_z)
                    / self.dz
                ),
            ]
        print("box_length=", self.length)
        print("box_width=", self.width)
        print("box_height=", self.height)
        print("start_point=", self.start_point)
        print("end_point=", self.end_point)

    def path_searching(self, start, end):
        """
        Search for a path from the start to the end using the A* algorithm.

        Args:
            start (ndarray): Starting position.
            end (ndarray): Target position.

        Returns:
            ndarray: The recovered path if found; None otherwise.
        """
        self.range_box_G(start=start, target=end)
        self.Astar = A_Search(
            self.start_point, self.end_point, self.Map, self.gripper_range
        )
        Result = self.Astar.process()
        if len(Result) > 0:
            Points_collection = []
            Recover = []
            for i in Result:
                print("path:(%d,%d,%d)" % (i.x, i.y, i.z))
                Points_collection.append([i.x, i.y, i.z])
                recover_point = [
                    self.dx * i.x + self.min_x,
                    self.dy * i.y + self.min_y,
                    self.dz * i.z + self.min_z,
                ]
                Recover.append(recover_point)
            Recover = np.array(Recover)
            Recover = Recover[::-1]
            Recover = np.append(start, Recover).reshape(-1, 3)
            Recover = np.append(Recover, end).reshape(-1, 3)

            return Recover
        else:
            print("The path is not found!")
            return None

    def recover_pos(self, point_collection):
        """
        Convert path points from voxelized space to real-world coordinates.

        Args:
            point_collection (list): List of voxelized points.

        Returns:
            list: List of points in real-world coordinates.
        """
        Recover = []
        for point in point_collection:
            recover_point = [
                self.dx * point[0] + self.min_x,
                self.dy * point[1] + self.min_y,
                self.dz * point[2] + self.min_z,
            ]
            Recover.append(recover_point)

        return Recover

    def path_smoothing(self, Path_points, t_final, freq):
        """
        Smooth the path using B-spline interpolation.

        Args:
            Path_points (list): List of path points.
            t_final (float): Total time for the motion.
            freq (float): Frequency of the motion.

        Returns:
            tuple: Smoothed x, y, and z coordinates.
        """
        self.n = int(t_final * freq)
        self.t = np.zeros(self.n)
        self.d = np.zeros((self.n, len(self.start_point)))
        self.d_dot = np.zeros((self.n, len(self.start_point)))
        self.d_ddot = np.zeros((self.n, len(self.start_point)))

        Path_points = np.array(Path_points)
        t = np.arange(len(Path_points))
        tck_x = splrep(t, Path_points[:, 0], k=2, s=0)
        tck_y = splrep(t, Path_points[:, 1], k=2, s=0)
        tck_z = splrep(t, Path_points[:, 2], k=2, s=0)
        t_new = np.linspace(0, len(Path_points) - 1, self.n)
        x_new = splev(t_new, tck_x)
        y_new = splev(t_new, tck_y)
        z_new = splev(t_new, tck_z)

        for i in range(self.n):
            self.t[i] = i * t_final / self.n
            self.d[i] = np.array([x_new[i], y_new[i], z_new[i]])
            if i == 0:
                self.d_dot[i] = np.array([0, 0, 0])
                self.d_ddot[i] = np.array([0, 0, 0])
            else:
                x_d = (x_new[i] - x_new[i - 1]) * freq
                y_d = (y_new[i] - y_new[i - 1]) * freq
                z_d = (z_new[i] - z_new[i - 1]) * freq
                self.d_dot[i] = np.array([x_d, y_d, z_d])
                x_dd = (self.d_dot[i, 0] - self.d_dot[i - 1, 0]) * freq
                y_dd = (self.d_dot[i, 1] - self.d_dot[i - 1, 1]) * freq
                z_dd = (self.d_dot[i, 2] - self.d_dot[i - 1, 2]) * freq
                self.d_ddot[i] = np.array([x_dd, y_dd, z_dd])

        return x_new, y_new, z_new


class point:
    """
    Class representing a point in 3D space, used for pathfinding.
    """

    _list = []
    _tag = True

    def __new__(cls, key):
        for i in point._list:
            if i.x == key[0] and i.y == key[1] and i.z == key[2]:
                point._tag = False
                return i
        nt = super(point, cls).__new__(cls)
        point._list.append(nt)
        return nt

    def __init__(self, key):
        x = key[0]
        y = key[1]
        z = key[2]
        if point._tag:
            self.x = x
            self.y = y
            self.z = z
            self.father = None
            self.F = 0
            self.G = 0
            self.cost = 0
        else:
            point._tag = True

    @classmethod
    def clear(cls):
        point._list = []

    def __eq__(self, T):
        if type(self) == type(T):
            return (self.x, self.y, self.z) == (T.x, T.y, T.z)
        else:
            return False

    def __str__(self):
        return "(%d,%d, %d)[F=%d,G=%d,cost=%d][father:(%s)]" % (
            self.x,
            self.y,
            self.z,
            self.F,
            self.G,
            self.cost,
            str((self.father.x, self.father.y)) if self.father != None else "null",
        )


class A_Search:
    """
    Class implementing the A* pathfinding algorithm.
    """

    def __init__(self, arg_start, arg_end, arg_map, gri_range):
        self.start = point(arg_start)
        self.end = point(arg_end)
        self.Map = arg_map
        self.gri_range = gri_range
        self.Map_scan = []
        self.open = []
        self.close = []
        self.result = []
        self.count = 0
        self.useTime = 0
        self.open.append(self.start)

    def cal_F(self, loc):
        G = loc.father.G + loc.cost
        H = self.getEstimate(loc)
        F = G + H
        return {"G": G, "H": H, "F": F}

    def F_Min(
        self,
    ):
        if len(self.open) <= 0:
            return None
        t = self.open[0]
        for i in self.open:
            if i.F < t.F:
                t = i
        return t

    def getAroundPoint(self, loc):
        nl = []
        print("Start to find the aroundPoint of the tar")

        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    x = loc.x + i
                    y = loc.y + j
                    z = loc.z + k

                    if (
                        x < 0
                        or x >= self.Map.shape[0]
                        or y < 0
                        or y >= self.Map.shape[1]
                        or z < 0
                        or z >= self.Map.shape[2]
                        or self.Map[x][y][z] == 1
                    ):
                        print("The (%d,%d,%d) has out of range" % (x, y, z))
                    elif i == j == k == 0:
                        print("Reach the father point")
                    else:
                        if (
                            self.Map[x - i][y][z]
                            + self.Map[x][y - j][z]
                            + self.Map[x][y][z - k]
                        ) == 0:
                            nt = point([x, y, z])
                            nt.cost = self.getFcost(i, j, k)
                            nl.append(nt)
                        else:
                            print("The (%d,%d,%d) is not reached" % (x, y, z))
        print("The aroundPoint have been gotten")

        return nl

    def getAroundPoint_G(self, loc):
        nl = []
        print("Start to find the aroundPoint of the tar")

        for i in [0, -1, 1]:
            for j in [0, -1, 1]:
                for k in [0, -1, 1]:
                    x = loc.x + i
                    y = loc.y + j
                    z = loc.z + k

                    max_x, min_x = min(
                        x + self.gri_range[0], self.Map.shape[0] - 1
                    ), max(x - self.gri_range[0], 0)
                    max_y, min_y = min(
                        y + self.gri_range[1], self.Map.shape[1] - 1
                    ), max(y - self.gri_range[1], 0)
                    max_z, min_z = min(
                        z + self.gri_range[2], self.Map.shape[2] - 1
                    ), max(z - self.gri_range[3], 0)
                    flag = 0

                    if (
                        x < 0
                        or x >= self.Map.shape[0]
                        or y < 0
                        or y >= self.Map.shape[1]
                        or z < 0
                        or z >= self.Map.shape[2]
                        or self.Map[x][y][z] == 1
                    ):
                        flag = 1
                        print("The (%d,%d,%d) has out of range" % (x, y, z))
                    elif i == j == k == 0:
                        flag = 1
                        print("Reach the father point")
                    else:
                        for x1 in range(min_x, max_x + 1, 1):
                            for y1 in range(min_y, max_y + 1, 1):
                                for z1 in range(min_z, max_z + 1, 1):
                                    if self.Map[x1][y1][z1] == 1 and flag == 0:
                                        flag = 1
                                        print(
                                            "The (%d,%d,%d) has out of range"
                                            % (x, y, z)
                                        )
                                        continue

                                    if (
                                        flag == 0
                                        and 0 <= (x1 - i) < self.Map.shape[0]
                                        and 0 <= (y1 - i) < self.Map.shape[1]
                                        and 0 <= (z1 - i) < self.Map.shape[2]
                                    ):
                                        if (
                                            self.Map[x1 - i][y1][z1]
                                            + self.Map[x1][y1 - j][z1]
                                            + self.Map[x1][y1][z1 - k]
                                        ) > 0:
                                            flag = 1
                                            print(
                                                "The (%d,%d,%d) is not reached"
                                                % (x, y, z)
                                            )
                                            continue

                    if flag == 0:
                        nt = point([x, y, z])
                        nt.cost = self.getFcost(i, j, k)
                        nl.append(nt)
        print("The aroundPoint have been gotten")

        return nl

    def addToOpen(self, l, father):
        for i in l:
            if i not in self.open:
                if i not in self.close:
                    i.father = father
                    self.open.append(i)
                    r = self.cal_F(i)
                    i.G = r["G"]
                    i.F = r["F"]
            else:
                tf = i.father
                i.father = father
                r = self.cal_F(i)
                if i.F > r["F"]:
                    i.G = r["G"]
                    i.F = r["F"]
                else:
                    i.father = tf

        print("The points have been addtoOpen")

    def getEstimate(self, loc):
        return (
            abs(loc.x - self.end.x) + abs(loc.y - self.end.y) + abs(loc.z - self.end.z)
        ) * 10

    def getFcost(self, dx, dy, dz):
        if abs(dx) + abs(dy) + abs(dz) == 1:
            cost = 10
        elif abs(dx) + abs(dy) + abs(dz) == 2:
            cost = 14
        elif abs(dx) + abs(dy) + abs(dz) == 3:
            cost = 17
        else:
            cost = 0

        return cost

    def process(self):
        while True:
            self.count += 1
            tar = self.F_Min()
            if tar == None:
                self.result = None
                self.count = -1
                print("The target is None")
                break
            else:
                print("tar=", tar)
                aroundP = self.getAroundPoint_G(tar)
                self.addToOpen(aroundP, tar)
                self.open.remove(tar)
                self.close.append(tar)
                if self.end in self.open:
                    e = self.end
                    self.result.append(e)
                    while True:
                        e = e.father
                        if e == None:
                            break
                        self.result.append(e)
                    point.clear()
                    break
        return self.result


if __name__ == "__main__":
    robots = "IIWA"
    env = HammerPlaceEnv(
        robots,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        render_camera="frontview",
        control_freq=20,
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    )

    env.reset()
    M = Motion_planning(env=env, dx=0.01, dy=0.01, dz=0.01)
    start = env._eef_xpos
    end = env._slide_handle_xpos
    env.close()
    Points_recover = M.path_searching(start=start, end=end)
    if Points_recover is not None:
        X, Y, Z = M.path_smoothing(Path_points=Points_recover, t_final=5, freq=20)
        print("path_points: ", Points_recover)
    else:
        print("The path is not found!!")
