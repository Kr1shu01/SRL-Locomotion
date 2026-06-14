from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi
import numpy as np
import torch
from humanoid.envs import LeggedRobot
from humanoid.utils import math

from humanoid.utils.terrain import HumanoidTerrain

import humanoid.utils.math


# from collections import deque

class x02jumpEnv(LeggedRobot):

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # cfg: 环境的配置对象，包含了机器人的结构、控制、奖励等相关信息。
        # sim_params: 仿真参数，用于设置仿真环境，如步长、重力等。
        # physics_engine: 使用的物理引擎（如 PhysX 或 Flex）。
        # sim_device: 仿真设备（CPU 或 GPU）。
        # headless: 是否以无头模式运行（不显示图形界面）。
        self.DTS = 0.066
        self.DSH = 0.02649
        self.LSH = 0.08585
        self.last_feet_z = 0.06
        # last_feet_z: 记录上次步态中脚部的 z 轴高度。
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        # 存储每个环境中双脚的高度信息。
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        # 重置环境索引。
        self.env_features = torch.zeros((self.num_envs, 5), device=self.device)
        # 用于存储环境特征和因素（如摩擦力、负载等）。
        # torch.zeros((4096, 2), device='cuda:0')会创建一个 4096 行 2 列的张量，初始值全部为 0，并且放在 GPU 的第一个设备上进行存储和运算。
        self.env_factors = torch.zeros((self.num_envs, 11), device=self.device)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos, device=self.device)
        self.ref_dof_vel = torch.zeros_like(self.dof_vel, device=self.device)
        self.ref_com_pos = torch.zeros_like(self.base_pos, device=self.device)
        self.ref_com_vel = torch.zeros_like(self.base_lin_vel, device=self.device)
        self.ref_feet_pos =torch.zeros_like(self.base_pos, device=self.device)
        self.has_jumped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.jump_timer = torch.zeros(self.num_envs, device=self.device)  # 添加一个跳跃计时器
        self.compute_observations()
        self.step_counter = 0
        self.speed_factor = 1.2  # 控制参考轨迹的速度
        self.jump_period = 0.4  # 跳跃周期
        self.jump_height_factor = 0.35  # 跳跃高度的影响因子
        # 膝关节的活动范围（弧度）
        self.knee_min_angle = -2.0071  # -115° 转换为弧度
        self.knee_max_angle = -0.0873  # -5° 转换为弧度
        # 记录前一帧的膝关节参考位置，用于平滑过渡
        self.last_p_knee = None
        #
        # 计算初始观测值。

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        # 用于对机器人施加随机的线速度和角速度，模拟外界干扰对机器人的影响，从而增强模型的鲁棒性。
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        # 允许的最大线速度干扰
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        # 允许的最大角速度干扰
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque
        # 更新机器人的线速度和角速度
        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))
        # 将更新的根状态应用到仿真中


    def asin(self, x):
        min = -1 * torch.ones(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        max = 1 * torch.ones(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        x = self.bnd(x, min, max)
        return torch.arcsin(x)

    def bnd(self, x, min_val, max_val):
        return torch.maximum(torch.minimum(x, max_val), min_val)

    def acos(self, x):
        min = -1 * torch.ones(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        max = 1 * torch.ones(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        x = self.bnd(x, min, max)
        return torch.arccos(x)

    def VlegIK(self, vh, vl, va):
        # 从虚拟腿到膝关节腿
        LGS = 0.35
        LGT = 0.35
        t2 = LGS * LGS
        t3 = LGT * LGT
        t4 = vl * vl
        t5 = 1.0 / LGS
        t6 = 1.0 / LGT
        t7 = -torch.pi
        t8 = 1.0 / vl
        t9 = -t2
        t10 = -t4
        t11 = t3 + t4 + t9
        t12 = t2 + t3 + t10
        t13 = (t5 * t6 * t12) / 2.0
        t14 = (t6 * t8 * t11) / 2.0
        t15 = self.acos(t13)
        t16 = self.acos(t14)
        jh = t16 + vh
        jk = t7 + t15
        ja = t7 + t15 + t16 + va
        # print([0, self.jst, jh, jk, ja])
        return [0, self.jst, jh, jk, ja]

    def setSwPt(self, px, py, pz):
        EPS = 1e-6  # 请根据实际情况调整这个值

        pxt = px  # = -0.0
        px -= 0.000
        pyt = py  # = 0.09249
        pzt = pz  # = 0.76585
        pat = pa = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)

        dy = py - self.DTS
        t = -(pz - torch.sqrt(-self.DSH * self.DSH + dy * dy + pz * pz)) / (self.DSH + dy)

        t[torch.abs(self.DSH + dy) < EPS] = -self.DSH / pz[torch.abs(self.DSH + dy) < EPS]

        js = 2 * torch.arctan(t)
        l1 = pz / torch.cos(js)
        l2 = l1 + self.DSH * torch.tan(js) - self.LSH
        ax = self.asin(torch.sin(pa) / torch.cos(js))  # cos(js) = sin(pa) / sin(ax);
        vl = torch.sqrt(px * px + l2 * l2)  # vl = min(vl, cfg['VLM'])
        vh = torch.arctan(px / l2)
        va = vh + ax
        self.jst = js
        return self.VlegIK(vh, vl, va)
    
    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = stance_mask[:, 0]
        # Double support phase
        # stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    # def compute_ref_state(self):
    #     phase = self._get_phase()
    #     py = torch.ones_like(phase) * 0.05
    #     px = torch.ones_like(phase) * 0.0 + 0.0 * abs(torch.sin(2 * torch.pi * phase))
    #     # pz = torch.ones_like(phase) * 0.7308 - 0.2 * abs(torch.sin(2 * torch.pi * phase))
    #     pz = torch.ones_like(phase) * 0.7308 - 0.2 * abs(torch.sin(2 * torch.pi * phase))
    #     # pz = torch.clip(pz, 0.7308, 1.2)
    #     [x, self.jst, jh, jk, ja] = self.setSwPt(px, py, pz)

    #     self.ref_dof_pos[:, 1] = 0.
    #     self.ref_dof_pos[:, 2] = jh
    #     self.ref_dof_pos[:, 3] = jk
    #     self.ref_dof_pos[:, 4] = -ja

    #     self.ref_dof_pos[:, 6] = 0.
    #     self.ref_dof_pos[:, 7] = self.ref_dof_pos[:, 2]
    #     self.ref_dof_pos[:, 8] = self.ref_dof_pos[:, 3]
    #     self.ref_dof_pos[:, 9] = self.ref_dof_pos[:, 4]

    #     self.ref_action = 1 * self.ref_dof_pos

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # 创建仿真环境，包括地形、机器人等
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
        # 创建环境实例

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise #是否添加噪声
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands  噪声向量，用于对不同的观测值添加噪声
        noise_vec[5: 15] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[15: 25] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[25: 28] = noise_scales.ang_vel * self.obs_scales.ang_vel  # ang vel
        noise_vec[28: 30] = noise_scales.quat * self.obs_scales.quat  # euler x,y
        noise_vec[30: 40] = 0
        return noise_vec

    def compute_observations(self):

        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 10
            self.dof_vel * self.obs_scales.dof_vel,  # 10
            self.actions,  # 10
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 18.,  # 1
            torch.zeros((self.num_envs, 2), device=self.device),  # 2
            torch.zeros((self.num_envs, 2), device=self.device),  # 2
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 12D
            dq,  # 12D
            self.actions,  # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.add_noise:
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def _reward_joint_pos(self):  # (关节位置奖励): 直接影响机器人的姿态。
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_action.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        error = torch.where(torch.abs(self.base_lin_vel[:, 2])>3.4, -2*torch.square(torch.abs(self.base_lin_vel[:, 2])-3.4), torch.square(self.base_lin_vel[:, 2]))
        return error
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        move = torch.sum(torch.square(self.base_ang_vel[:, :3]), dim=1)
        move += torch.sum(torch.square(self.base_lin_vel[:, :3]), dim=1)
        move += torch.sum(torch.square(self.dof_vel), dim=1)
        return move * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def _reward_base_height(self):
        # Penalize base height away from target
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        condition = sin_pos > 0
        base_height = self.root_states[:, 2] - self.measured_heights
        # 计算 error
        error_true = torch.square(0.96 + 0.4*sin_pos - base_height)
        error_false = torch.square(0.7 - base_height)
        error = torch.where(condition, error_true, error_false)
        return torch.exp(-error/self.cfg.rewards.tracking_sigma)
    
    def _reward_no_fly(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.where(sin_pos > 0,
                                    torch.sum(1. * contacts, dim=1) == 0,
                                    torch.sum(1. * contacts, dim=1) == 2)
        error = 1. * single_contact.float()  
        return error

    
    def _reward_sym(self):
        indices1 = [0, 2, 3, 4]
        indices2 = [5, 7, 8, 9]

        selected_dof_pos_1 = self.dof_pos[:, indices1]
        selected_dof_pos_2 = self.dof_pos[:, indices2]

        # 计算平方差之和
        lin_vel_error = torch.sum(torch.square(selected_dof_pos_1 - selected_dof_pos_2), dim=1)
        # lin_vel_error = torch.square(torch.abs(self.dof_pos[:, 2]))
        lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 1]))
        lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 6]))
        # lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 11]))

        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_foot_height(self):
        stance_mask = (self.episode_length_buf < 35) & (self.episode_length_buf != 0)
        left_foot_height = self.rigid_state[:, self.feet_indices[0], 2]  # 左脚的高度
        right_foot_height = self.rigid_state[:, self.feet_indices[1], 2]  # 右脚的高度
        foot_heights = (left_foot_height + right_foot_height) / 2  # 计算平均脚高
        foot_distance_from_ground = torch.abs(foot_heights - self.ref_feet_pos[:, 2])
        reward = torch.exp(-foot_distance_from_ground * 100)  # 脚部高度接近参考值时奖励更高
        return reward

    def _reward_stand_still(self):
        # penalize motion at zero commands
        mask = (self.episode_length_buf >= 60) & (self.episode_length_buf != 0)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(mask, r.clone(),
                        torch.zeros_like(r))
        return r


    def _reward_track_joint(self):
        position_difference = abs(self.dof_pos[:, :5] - self.dof_pos[:, 5:10])
        reward = -position_difference  # 负数表示我们希望差异最小化
        target_difference = position_difference / 0.001
        target_reward = torch.clamp(1 - target_difference, min=0)
        # print(self.ref_dof_pos[1, :5],self.dof_pos[1, :5])
        threshold = 0.1
        reward = torch.sum(reward, dim=1)
        reward[abs(reward) < threshold] = 1.0
        reward[abs(self.dof_pos[:, 0]) > 0.1] = -1
        reward[abs(self.dof_pos[:, 5]) > 0.1] = -1
        reward[abs(self.dof_pos[:, 1]) > 0.2] = -1
        reward[abs(self.dof_pos[:, 6]) > 0.2] = -1
        reward[(self.dof_pos[:, 1]) < 0.0] = -1000
        reward[(self.dof_pos[:, 6]) < 0.0] = -1000
        # reward =torch.sum(target_reward, dim=1)/5
        # reward[(self.episode_length_buf < 10) | (self.episode_length_buf > 35)] = 0
        return reward

    def _reward_torques(self):
        # 使用尽量少的扭矩，节省能耗
        return torch.exp(-torch.sum(torch.square(self.torques), dim=1) * 0.05)

    def _reward_action_smoothness(self):
        # 动作平滑，减少突变
        # action_diff = torch.sum((self.actions - self.last_actions) ** 2, dim=1)
        # return torch.exp(-action_diff * 3)
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.1 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_feet_distance(self):  # (脚距离奖励): 保持合适的脚间距对于稳定性和步态非常重要。
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):  # (膝盖距离奖励): 保持合适的膝盖间距对运动协调性有帮助。
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_joint_vel(self):
        joint_vel = self.dof_vel.clone()
        vel_target = self.ref_dof_vel.clone()
        diff = joint_vel - vel_target
        return torch.exp(-2 * torch.norm(diff, dim=1))  # 使用指数函数奖励关节速度跟踪

    def _reward_base_vel(self):  # (关节位置奖励): 直接影响机器人的姿态。
        base_vel_target = self.ref_com_vel[:, 2]
        base_vel = self.base_lin_vel[:, 2]
        diff = base_vel_target - base_vel
        # mse = torch.mean((joint_vel - vel_target) ** 2, dim=1)
        # r1 = torch.exp(-20 * mse)
        # return -2 * torch.norm(diff, dim=1)
        # r1 = torch.exp(-1 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.8)
        lin_vel_error = torch.square(diff)
        # 当速度超过目标速度时给予额外奖励，鼓励更快的起跳
        tracking_sigma = 5  # 由原来的 3 调整为 5，增加对速度偏差的敏感性
        vel_reward = torch.where(base_vel > base_vel_target,
                                 torch.exp(-lin_vel_error * tracking_sigma) + 0.5,
                                 torch.exp(-lin_vel_error * tracking_sigma))

        return vel_reward

    def _reward_com_height(self):

        com_height = self.ref_com_pos[:, 2]
        measured_heights = torch.mean(self.rigid_state[:, self.feet_indices, 2])
        com_height_base = self.root_states[:, 2] - (measured_heights - 0.06)
        height_diff = com_height - com_height_base
        # return torch.exp(-height_diff * 15)  # Increase factor to penalize deviation from target
        r2 = torch.exp(-0.3 * height_diff) - 0.5 * height_diff.clamp(0, 0.4)
        # r2 = torch.exp(-0.2 * height_diff) + torch.where(com_height_base > 0.5458, 0.5, 0.0)
        return r2

    def _reward_jump_height(self):
        # jump_target_height = (self.ref_com_vel[:, 2] * self.ref_com_vel[:, 2]) /(2*9.8)
        jump_target_height = 1.2
        current_height = self.root_states[:, 2]
        jump_diff = jump_target_height - current_height
        r3 = torch.exp(-0.2 * torch.abs(jump_diff))
        # height_reward = torch.where(current_height >= jump_target_height, 1.0, current_height * 0.8)0
        # height_reward = torch.exp(-0.3 * torch.abs(jump_diff))
        # total_reward = r3 + height_reward
        return r3

    def _reward_knee(self):
        # 奖励脚离地高度 - 鼓励更高的跳跃
        knee_left_pos = self.ref_dof_pos[:, 3]  # 左膝关节
        knee_right_pos = self.ref_dof_pos[:, 8]  # 右膝关节
        max_knee_extension = torch.max(knee_left_pos, knee_right_pos)
        return torch.exp(max_knee_extension * 1.0)  # 给予高度奖励

    def _reward_soft_landing(self):
        # 奖励软着陆 - 惩罚较大的着陆冲击力
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        landing_impact = torch.norm(contact_force, dim=1)
        return torch.exp(-landing_impact * 500)  # 给予轻柔着陆奖励

    def _reward_contact(self):
        """
        Reward for both feet contact with ground upon landing.
        """
        feet_contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        contact_reward = torch.all(feet_contact, dim=1).float()
        return contact_reward

    def _reward_balance(self):
        lin_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return torch.exp(-lin_vel * 0.1)

    def _reward_collision(self):  # 鼓励机器人在跳跃时尽可能减少接触地面
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                     dim=1)

    def _reward_feet_contact_number(self):  # (脚接触次数奖励): 奖励与步态相符的接触次数，有助于步态控制。
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase.
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

# python scripts/train.py --task=x02_ppo_jump --run_name=v1 --headless --num_envs=4096
# python scripts/play.py --task=x02_ppo_jump --run_name v1 --num_envs 64
# python scripts/sim2sim.py --load_model ../logs/x02_ppo_jump/exported/policies/policy_1.pt
