from collections import OrderedDict
import numpy as np
from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject

from robosuite.models.objects import HammerObject, DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer
import robosuite.utils.transform_utils as T

from robosuite_task_zoo.models.hammer_place import CabinetObject


class HammerPlaceEnv(SingleArmEnv):
    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            use_latch=False,# 这里注意lock = use_latch = False
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            contact_threshold=2.0
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (1.2, 1.2, 0.05)
        self.table_offset = (0, 0, 0.80)

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None

        self.objects = []

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward`
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos([-0.2, 0, 0])

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.7386131746834771, -4.392035683362857e-09, 0.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.03, 0.018, 0.025]

        # initialize objects of interest
        self.door = DoorObject(
            name="Door",
            friction=None,
            damping=None,
            lock=self.use_latch,
        )
        # door = self.door.get_obj()
        # door.set('pos', array_to_string((0.2, -0.40, 0.03)))
        # mujoco_arena.table_body.append(door)

        cabinet_pos = np.array([np.random.rand() * (0.4 - 0.2) + 0.3, np.random.rand() * (0.4 - 0.3) + 0.3, 0.03])
        self.cabinet_object = CabinetObject(
            name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj();
        cabinet_object.set("pos", array_to_string(cabinet_pos));
        mujoco_arena.table_body.append(cabinet_object)
        self.cabinet_object._set_cabinet_damping(200)
        for obj_body in [
            self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-door",
                mujoco_objects=self.door,
                x_range=[0.4, 0.3],
                y_range=[-0.4, -0.2],
                rotation=(-np.pi / 2, -np.pi / 2),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            ))

        # self.placement_initializer.append_sampler(
        #     sampler=UniformRandomSampler(
        #         name="ObjectSampler-cabinet",
        #         mujoco_objects=self.cabinet_object,
        #         x_range=[0.3, 0.4],
        #         y_range=[0, 0.3],
        #         rotation=(0, 0),
        #         rotation_axis='z',
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=self.table_offset,
        #         z_offset=0.03,
        #     ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.door
        )

        # self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)
        # self.model.merge_assets(self.door)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.door.latch_body)
        self.object_body_ids['cabinet'] = self.sim.model.body_name2id(self.cabinet_object.cabinet_base)
        self.object_body_ids['cabinet_drawer_link'] = self.sim.model.body_name2id(
            self.cabinet_object.cabinet_drawer_link)
        # 增加关节信息的索引id，从而获取关节位置
        self.object_joint_ids = dict()
        self.object_joint_ids["cabinet_joint"] = self.sim.model.joint_name2id(self.cabinet_object.cabint_joint)
        self.object_joint_ids["hinge_joint"] = self.sim.model.joint_name2id(self.door.hinge_joint)

        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.cabinet_object_site_id = self.sim.model.site_name2id(self.cabinet_object.important_sites['slide_handle'])
        self.cabinet_base_center_site_id = self.sim.model.site_name2id(
            self.cabinet_object.important_sites['base_center'])

        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])
        self.cabinet_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[1])

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # Define sensor callbacks
            @sensor(modality=modality)
            def door_pos(obs_cache):
                return self.door_pos

            @sensor(modality=modality)
            def cabinet_pos(obs_cache):
                return self.cabinet_pos

            @sensor(modality=modality)
            def handle_pos(obs_cache):
                return self._handle_xpos

            @sensor(modality=modality)
            def door_to_eef_pos(obs_cache):
                return (
                    obs_cache["door_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "door_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def handle_to_eef_pos(obs_cache):
                return (
                    obs_cache["handle_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "handle_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def hinge_qpos(obs_cache):
                return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            sensors = [door_pos, cabinet_pos, handle_pos, door_to_eef_pos, handle_to_eef_pos, hinge_qpos]
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:
                @sensor(modality=modality)
                def handle_qpos(obs_cache):
                    return np.array([self.sim.data.qpos[self.handle_qpos_addr]])

                sensors.append(handle_qpos)
                names.append("handle_qpos")

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # We know we're only setting a single object (the door), so specifically set its pose
            # 这个主要是从采样器中获取门的位置和姿态信息，并对相关信息进行更新
            door_pos, door_quat, _ = object_placements[self.door.name]
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

            # cabinet_pos, cabinet_quat, _ = object_placements[self.cabinet_object.name]
            # cabinet_body_id = self.sim.model.body_name2id(self.cabinet_object.root_body)
            # self.sim.model.body_pos[cabinet_body_id] = cabinet_pos
            # self.sim.model.body_quat[cabinet_body_id] = cabinet_quat

    def _check_success(self):
        """
        Check if door has been opened.

        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        return hinge_qpos > 0.3

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper, target=self.door.important_sites["handle"], target_type="site"
            )

    @property
    def _handle_xpos(self):
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def door_pos(self):
        return np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])

    @property
    def cabinet_pos(self):
        return np.array(self.sim.data.body_xpos[self.object_body_ids["cabinet"]])

    @property
    def _slide_handle_xpos(self):
        return self.sim.data.site_xpos[self.cabinet_object_site_id]

    @property
    def _cabinet_base_center_xpos(self):
        return self.sim.data.site_xpos[self.cabinet_base_center_site_id]

    @property
    def _gripper_to_handle(self):
        return self._handle_xpos - self._eef_xpos

    @property
    def cabinet_joint_pos(self):
        return self.sim.data.body_xpos[self.cabinet_qpos_addrs]
        # return np.array(self.sim.data.body_xpos[self.object_joint_ids["cabinet_joint"]])

    @property
    def door_joint_pos_1(self):
        return np.array(self.sim.data.qpos[self.hinge_qpos_addr])
        # return np.array(self.sim.data.body_xpos[self.object_joint_ids["hinge_joint"]])

    # @property
    # def door_joint_pos(self):
    #     return np.array(self.sim.data.body_xpos[self.object_joint_ids["hinge_joint"]])

    @property
    def door_frame_pos(self):
        return np.array(self.sim.data.body_xpos[self.object_body_ids["frame"]])

    @property
    def door_latch_pos(self):
        return np.array(self.sim.data.body_xpos[self.object_body_ids["latch"]])

