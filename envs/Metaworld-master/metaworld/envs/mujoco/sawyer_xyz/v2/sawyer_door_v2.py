from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import RenderMode, SawyerXYZEnv
from metaworld.envs.mujoco.utils import reward_utils
from metaworld.types import InitConfigDict


class SawyerDoorEnvV2(SawyerXYZEnv):
    def __init__(
        self,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)
        goal_low = (-0.3, 0.4, 0.1499)
        goal_high = (-0.2, 0.5, 0.1501)

        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config: InitConfigDict = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.1, 0.95, 0.15]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.2, 0.7, 0.15])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.door_qpos_adr = self.model.joint("doorjoint").qposadr.item()
        self.door_qvel_adr = self.model.joint("doorjoint").dofadr.item()

        self._random_reset_space = Box(
            np.array(obj_low), np.array(obj_high), dtype=np.float64
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self) -> str:
        return full_v2_path_for("sawyer_xyz/sawyer_door_pull.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        assert self._target_pos is not None
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
        ) = self.compute_reward(action, obs)

        success = float(abs(obs[4] - self._target_pos[0]) <= 0.08)

        info = {
            "success": success,
            "near_object": reward_ready,
            "grasp_success": reward_grab >= 0.5,
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": 0,
            "unscaled_reward": reward,
        }

        return reward, info

    @property
    def _target_site_config(self) -> list[tuple[str, npt.NDArray[Any]]]:
        return []

    def _get_pos_objects(self) -> npt.NDArray[Any]:
        return self.data.geom("handle").xpos.copy()

    def _get_quat_objects(self) -> npt.NDArray[Any]:
        return Rotation.from_matrix(
            self.data.geom("handle").xmat.reshape(3, 3)
        ).as_quat()

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_qpos_adr] = pos
        qvel[self.door_qvel_adr] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self) -> npt.NDArray[np.float64]:
        self._reset_hand()
        self.objHeight = self.data.geom("handle").xpos[2]

        self.obj_init_pos = self._get_state_rand_vec()
        self._target_pos = self.obj_init_pos + np.array([-0.3, -0.45, 0.0])

        self.model.body("door").pos = self.obj_init_pos
        self.model.site("goal").pos = self._target_pos
        self._set_obj_xyz(np.array(0))
        assert self._target_pos is not None
        self.maxPullDist = np.linalg.norm(
            self.data.geom("handle").xpos[:-1] - self._target_pos[:-1]
        )
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2
        self.model.site("goal").pos = self._target_pos
        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions: npt.NDArray[Any]) -> float:
        return float((np.clip(actions[3], -1, 1) + 1.0) / 2.0)

    @staticmethod
    def _reward_pos(obs: npt.NDArray[Any], theta: float) -> tuple[float, float]:
        hand = obs[:3]
        door = obs[4:7] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid="long_tail",
            )
        )
        # move the hand to a position between the handle and the main door body
        in_place = reward_utils.tolerance(
            float(np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01]))),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.2  # Relative importance of just *trying* to open the door at all
        b = 0.8  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi / 90.0) + b * reward_utils.tolerance(
            np.pi / 2.0 + np.pi / 6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi / 3.0,
            sigmoid="long_tail",
        )

        return ready_to_open, opened

    def compute_reward(
        self, actions: npt.NDArray[Any], obs: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float]:
        assert (
            self._target_pos is not None
        ), "`reset_model()` must be called before `compute_reward()`."
        theta = float(self.data.joint("doorjoint").qpos.item())

        reward_grab = SawyerDoorEnvV2._reward_grab_effort(actions)
        reward_steps = SawyerDoorEnvV2._reward_pos(obs, theta)

        reward = sum(
            (
                2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
                8.0 * reward_steps[1],
            )
        )

        # Override reward on success flag
        if abs(obs[4] - self._target_pos[0]) <= 0.08:
            reward = 10.0

        return (
            reward,
            reward_grab,
            *reward_steps,
        )
