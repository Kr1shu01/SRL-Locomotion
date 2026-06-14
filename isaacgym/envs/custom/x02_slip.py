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

class x02slipEnv(LeggedRobot):

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
        self.feet = self.rigid_state[:, self.feet_indices, :3].clone()
        # self.last_feet_z = 0.06
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
        self.previous_rootx = self.root_states[:, 0]
        self.previous_rooty = self.root_states[:, 1]

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
        phase = (self.episode_length_buf) * self.dt
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = self.jump_zk < -0.1
        stance_mask[:, 1] = stance_mask[:, 0]
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        t = phase % 1.98
        py = torch.ones_like(t) * 0.05
        px = torch.ones_like(t) * 0.0 + 0.0 * abs(torch.sin(2 * torch.pi * t))
        pz = torch.ones_like(t) * 0.7 - 0.0 * (torch.cos(2 * torch.pi * t))

        px1 = 0.79687 * t**3 - 0.22761 * t **2 + 0.02091 * t + 0.01962
        px2 = -2.99073 * (t - 0.4)**3 + 2.83325 * (t - 0.4)**2 + 0.21099 * (t - 0.4) + 0.04366
        px3 = -9.69247 * (t - 0.6)**3 + 8.61694 * (t - 0.6)**2 - 2.64865 * (t - 0.6) + 0.12277
        px4 = -3.97864 * (t - 1.0)**3 - 1.0657 * (t - 1.0)**2 + 0.94157 * (t - 1.0) - 0.16949
        px5 = 0.12064 * (t - 1.19)**3 - 0.2467 * (t - 1.19)**2 + 0.18683 * (t - 1.19) - 0.05559

        pz1 = 11.05639 * t **3 - 6.56588 * t **2 + 0.04302 * t + 0.6992
        pz2 = -29.49624 * (t - 0.4)**3 + 14.52348 * (t - 0.4)**2 - 0.20425 * (t - 0.4) + 0.37244
        pz3 = -12.8121 * (t - 0.6)**4 + 11.70907 * (t - 0.6)**3 - 3.46796 * (t - 0.6)**2 + 0.19291 * (t - 0.6) + 0.71137
        pz4 = 29.29793 * (t - 1.0)**3 - 2.14202 * (t - 1.0)**2 - 2.0858 * (t - 1.0) + 0.64868
        pz5 = 0.4464 * (t - 1.19)**4 - 0.16473 * (t - 1.19)**3 - 0.89593 * (t - 1.19) **2 + 0.99609 * (t - 1.19) + 0.36358

        pz = torch.where(t > 1.19, pz5, 
            torch.where(t > 1.0, pz4, 
            torch.where(t > 0.6, pz3, 
            torch.where(t > 0.4, pz2, pz1)))) + 0.12
            
        px = torch.where(t > 1.19, px5, 
            torch.where(t > 1.0, px4, 
            torch.where(t > 0.6, px3, 
            torch.where(t > 0.4, px2, px1))))* 0.5
        
        [x, self.jst, jh, jk, ja] = self.setSwPt(px, py, pz)

        self.jump_zh = jh
        self.jump_zk = jk

        self.jump_za = -ja
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        # left foot stance phase set to default joint pos
        self.ref_dof_pos[:, 1] = 0.0
        self.ref_dof_pos[:, 2] = self.jump_zh
        self.ref_dof_pos[:, 3] = self.jump_zk
        self.ref_dof_pos[:, 4] = self.jump_za
        # right foot stance phase set to default joint pos
        self.ref_dof_pos[:, 6] = 0.0
        self.ref_dof_pos[:, 7] = self.jump_zh
        self.ref_dof_pos[:, 8] = self.jump_zk
        self.ref_dof_pos[:, 9] = self.jump_za

        self.ref_action = 1 * self.ref_dof_pos

        
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
        # 生成一个用于添加到观测值的噪声向量
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

    # def _get_env_factors(self):
    #     # 主要功能是根据给定的配置（cfg），将环境中的一些因素（如质心偏移、摩擦系数、电机强度等）标准化为特定的尺度，并返回这些标准化的环境因素。这些环境因素在训练中通常用于特权观测或用于调整策略模型的决策。
    #     # 质心    3
    #     com_displacements_scale, com_displacements_shift = math.get_scale_shift(self.cfg.normalization.com_displacement_range)
    #     com_displacements = (self.com_displacements - com_displacements_shift) * com_displacements_scale
    #
    #     # 摩擦系数    1
    #     friction_coeffs_scale, friction_coeffs_shift = math.get_scale_shift(self.cfg.normalization.friction_range)
    #     friction_coeffs = (self.friction_coeffs - friction_coeffs_shift) * friction_coeffs_scale
    #
    #     # 摩擦补偿      4
    #     restitutions_scale, restitutions_shift = math.get_scale_shift(self.cfg.domain_rand.restitution_range)
    #     restitutions = (self.restitutions - restitutions_shift) * restitutions_scale
    #
    #     # 电机强度      10
    #     motor_strengths_scale, motor_strengths_shift = math.get_scale_shift(self.cfg.normalization.motor_strength_range)
    #     motor_strengths = (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale
    #
    #     # 电机偏移量     10
    #     motor_offset_scale, motor_offset_shift = math.get_scale_shift(self.cfg.normalization.motor_offset_range)
    #     motor_offsets = (self.motor_offsets - motor_offset_shift) * motor_offset_scale
    #
    #     # 负载      1
    #     payloads_scale, payloads_shift = math.get_scale_shift(self.cfg.normalization.added_mass_range)
    #     payloads = (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale
    #     return com_displacements, friction_coeffs, restitutions, motor_strengths, motor_offsets, payloads

    def step(self, actions):
        # self.jump_timer += self.dt  # 计时器随着每个时间步增加
        # if torch.all(self.has_jumped) and torch.any(self.jump_timer > 0.02):  # 跳跃完成后0.02秒重置
        #     self.has_jumped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        #     self.jump_timer[:] = 0  # 重置计时器

        # 原始的动态随机化处理动作逻辑
        self.compute_ref_state()
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # print(self.ref_action)
        # 动作延迟随机化处理
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions

        # 调用父类的 step 方法，执行环境状态更新
        return super().step(actions)

    def compute_observations(self):

        phase = self._get_phase()
        self.compute_ref_state()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        diff = self.dof_pos - self.ref_dof_pos
        # print(diff)
        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 10
            self.dof_vel * self.obs_scales.dof_vel,  # 10
            self.actions,  # 10
            diff,  # 10 
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
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

    def get_target_velocity(self):  
        phase = self._get_phase()
        t = phase % 1.98
        vx = torch.ones_like(t) * 0.0
        vx1 = 5.41682 * t ** 3 - 0.36654 * t **2 + 0.0312 * t - 0.0005
        vx2 = -80.54291 * (t - 0.41)**3 + 12.09768 * (t - 0.41)**2 + 3.90848 * (t - 0.41) + 0.32456
        vx3 = torch.ones_like(t) * 0.946588
        vx4 = 76.30915 * (t - 1.0)**3 - 26.22602 * (t - 1.0)**2 - 0.33936 * (t - 1.0) + 0.94734
        vx5 =  0.385 * torch.exp(-13.056 * (t - 1.19))
            
        vx = torch.where(t > 1.19, vx5, 
            torch.where(t > 1.00, vx4, 
            torch.where(t > 0.60, vx3, 
            torch.where(t > 0.41, vx2, vx1))))
        
        vz = torch.ones_like(t) * 0.0
        vz1 = 1.18999 * t **3 + 30.88172 * t **2 - 12.73049 * t + 0.05268
        vz2 = -250.37119 * (t - 0.41)**3 - 20.73337 * (t - 0.41)**2 + 23.1536 * (t - 0.41) + 0.01772
        vz3 = -9.81 * (t- 0.6) + 1.94436
        vz4 = -206.50273 * (t - 1.0)**3 + 143.54032 * (t - 1.0)**2 - 8.8353 * (t - 1.0) - 1.97791
        vz5 = 0.142 + 0.633 * torch.exp(-20.000 * ((t- 1.19) - 0.113) ** 2)
        
        vz = torch.where(t > 1.19, vz5, 
            torch.where(t > 1.00, vz4, 
            torch.where(t > 0.60, vz3, 
            torch.where(t > 0.41, vz2, vz1))))
        
        return vx
#/////////////////////////////////////////////////////////////////rewards///////////////////////////////////////////////////////////////////////////
    
    def _reward_vxvz_tracking(self):
        target_vx = self.get_target_velocity()
        diff_vx = torch.sum(torch.square( target_vx- self.base_lin_vel[:, 0]))
        # diff_vz = torch.sum(torch.square( target_vz- self.base_lin_vel[:, 2]))
        diff_v =  diff_vx
        r = torch.exp(-0.00005 * torch.norm(diff_v))
        return r
    
    def _reward_joint_pos(self):  # (关节位置奖励)
        joint_pos = self.dof_pos.clone()
        pos_target = self.ref_action.clone()
        diff = joint_pos - pos_target
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        return r
    
    def _reward_jump_in_place(self):
        current_rootx = self.root_states[:, 0]
        current_rooty = self.root_states[:, 1]
        previous_rootx = self.previous_rootx
        previous_rooty = self.previous_rooty
        diff_x = torch.abs(current_rootx - previous_rootx)
        diff_y = torch.abs(current_rooty - previous_rooty)
        rx = torch.where(diff_x  < 0.001, 1.0, -1.0)  # 位置没变化时奖励为 1，否则惩罚为 -1
        ry = torch.where(diff_y  < 0.001, 1.0, -1.0) 
        self.previous_rootx = current_rootx.clone()
        self.previous_rooty = current_rooty.clone()
        return (rx + ry)/2

    def _reward_jump_forward(self):
        current_rootx = self.root_states[:, 0]  # 当前时间步的质心 x 坐标
        previous_rootx = self.previous_rootx  # 上一个时间步的质心 x 坐标
        r = torch.where(current_rootx > previous_rootx, 1, -0.5)

        self.previous_rootx = current_rootx.clone()
        return r
    
    def _reward_sym(self):

        # position_difference = abs(self.dof_pos[:, :5] - self.dof_pos[:, 5:10])
        # r = -torch.sum(position_difference ** 20, dim=1)
        # return r
        indices1 = [2, 3, 4]#索引腿部关节
        indices2 = [7, 8, 9]
        selected_dof_pos_1 = self.dof_pos[:, indices1]
        selected_dof_pos_2 = self.dof_pos[:, indices2]
        lin_vel_error = torch.sum(torch.square(selected_dof_pos_1 - selected_dof_pos_2), dim=1)
        # lin_vel_error = torch.square(torch.abs(self.dof_pos[:, 2]))
        # lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 1]))
        # lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 6])) #限制为0
        # lin_vel_error += torch.square(torch.abs(self.dof_pos[:, 7]))
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma )
        
    
    def _reward_fly(self):
        phase = self._get_phase()
        t = phase % 1.98
        fly_period = ((t >= 0.6) & (t <= 1.0)) # 返回布尔类型
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        contact_count = torch.sum(1. * contacts, dim=1)  # 统计接触的足部数量
        r = torch.where(fly_period, 
                        (contact_count == 0).float(),
                        (contact_count == 2).float())
        return r
    
    def _reward_feet_contact_number(self):  # (脚接触次数奖励): 奖励与步态相符的接触次数，有助于步态控制。

        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_base_height(self):
         # Penalize base height away from target
         phase = self._get_phase()
         sin_pos = torch.sin(2 * torch.pi * phase)
         condition = sin_pos > 0
         base_height = self.root_states[:, 2] - self.measured_heights
         # 计算 error
         error_true = torch.square(0.7 + 0.4*sin_pos - base_height)
         error_false = torch.square(0.46- base_height)
         error = torch.where(condition, error_true, error_false)
         return torch.exp(-error/self.cfg.rewards.tracking_sigma)

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
        mask = (self.episode_length_buf >= 198 * 4) & (self.episode_length_buf != 0)
        r = torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1))
        r = torch.where(mask, r.clone(),
                        torch.zeros_like(r))
        return r

    def _reward_lin_vel_z(self):
        mask = (self.episode_length_buf < 198) & (self.episode_length_buf != 0)
        r = torch.clamp(torch.abs(self.base_lin_vel[:, 2]), max=3.0)
        r = torch.where(mask, r.clone(),torch.zeros_like(r))
        return r
    
    def _reward_lin_vel_x(self):
        mask = (self.episode_length_buf < 198) & (self.episode_length_buf != 0)
        r = torch.relu(torch.clamp(self.base_lin_vel[:, 0], max=3.0))
        r = torch.where(mask, r.clone(), torch.zeros_like(r))
        return r

    def _reward_track_joint(self):
        position_difference = abs(self.dof_pos[:, :5] - self.dof_pos[:, 5:10])
        reward = -position_difference  # 负数表示我们希望差异最小化
        #target_difference = position_difference / 0.001
        #target_reward = torch.clamp(1 - target_difference, min=0)
        # print(self.ref_dof_pos[1, :5],self.dof_pos[1, :5])
        #threshold = 0.1
        reward = torch.sum(reward, dim=1)
        #reward[abs(reward) < threshold] = 1.0
        #reward[abs(self.dof_pos[:, 0]) > 0.1] = -1
        #reward[abs(self.dof_pos[:, 5]) > 0.1] = -1
        reward[abs(self.dof_pos[:, 1]) > 0.2] = -100
        reward[abs(self.dof_pos[:, 6]) > 0.2] = -100
        reward[(self.dof_pos[:, 1]) < 0.0] = -1000
        reward[(self.dof_pos[:, 6]) < 0.0] = -1000
        # reward =torch.sum(target_reward, dim=1)/5
        reward[(self.episode_length_buf < 10) | (self.episode_length_buf > 35)] = 0
        return reward

    def _reward_orientation(self):
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :3]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        rew = (quat_mismatch + orientation) / 2.
        # rew[torch.abs(self.base_euler_xyz[:, 1]) > 1.0] = -100
        # rew[self.ref_dof_pos[:, 3] > -0.1] = -100
        # rew[self.ref_dof_pos[:, 8] > -0.1] = -100
        rew[abs(self.dof_pos[:, 0]) > 0.1] = -1
        rew[abs(self.dof_pos[:, 5]) > 0.1] = -1
        rew[self.dof_pos[:, 1] > 0.1] = -1
        rew[self.dof_pos[:, 6] > 0.1] = -1
        rew[self.dof_pos[:, 1] < -0.1] = -1
        rew[self.dof_pos[:, 6] < -0.1] = -1

        return rew
    
    def _reward_pitch(self):
        pitch = self.base_euler_xyz[:, 1]
        r = torch.where(pitch < 0, torch.exp(pitch * 500),1)
        # r = torch.where(pitch < 0, torch.exp(pitch * 500), 
        #                  torch.where(pitch > 0, torch.exp(pitch * (-400)), 1)) 
     
        return r

    def _reward_torques(self):
        return torch.exp(-torch.sum(torch.square(self.torques), dim=1) * 0.05)

    def _reward_action_smoothness(self):
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.1 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
        rew[self._reward_joint_pos() < 0.6] = 0
        return rew

    def _reward_feet_distance(self):
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
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

    def _reward_jump_lift(self):
        # contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # # 获取脚部z轴位置并计算变化
        # feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.06 # 当前脚的高度
        # delta_z = feet_z - self.last_feet_z  # 跳跃过程中的高度变化
        # self.feet_height += delta_z
        # self.feet_height_left = self.feet_height[:, 0].unsqueeze(1)
        # self.feet_height_righ = self.feet_height[:, 1].unsqueeze(1)
        # self.feet_height_mean = (self.feet_height_left + self.feet_height_righ) * 0.5
        # self.last_feet_z = feet_z
        # # 目标离地高度奖励
        # jump_height_target = self.ref_com_pos[:, 2] + 0.1
        # jump_height = self.base_pos[:, 2] + self.feet_height_mean
        # measured_heights = torch.sum(
        #     self.rigid_state[:, self.feet_indices, 2], dim=1)
        # jump_height = self.root_states[:, 2] - (measured_heights - 0.06) + self.feet_height_mean
        # jump_diff = jump_height - jump_height_target
        # # jump_reward = torch.abs(jump_diff) < 0.01  # 使用阈值比较接近程度
        # # jump_reward = jump_reward.float().sum()  # 累计获得奖励
        # jump_reward = torch.exp(-1 * torch.norm(jump_diff, dim=1)) - 0.3 * torch.norm(jump_diff, dim=1).clamp(0, 0.1)
        # self.feet_height *= ~contact


        # measured_heights = torch.mean(self.rigid_state[:, self.feet_indices, 2]) - 0.06
        # # print(measured_heights)
        # jump_height_target =  self.ref_com_pos[:, 2] + measured_heights
        # jump_height = self.root_states[:, 2]
        # jump_diff = jump_height - jump_height_target
        # jump_reward = torch.exp(-2 * torch.norm(jump_diff)) - 0.3 * torch.norm(jump_diff).clamp(0, 0.3)
        # # print(jump_reward)
        # # jump_reward[measured_heights < 0] = 0
        # # print(jump_reward)
        # return jump_reward

        # 不动
        # jump_height = self.root_states[:, 2] - self.ref_com_pos[:, 2]
        # measured_heights = torch.mean(self.rigid_state[:, self.feet_indices, 2]) - 0.06
        # jump_diff = jump_height - measured_heights
        # jump_reward = torch.exp(-0.5 * torch.abs(jump_diff))
        # jump_reward += 0.5 * torch.abs(jump_diff).clamp(0, 0.5)
        # height_reward = torch.relu(jump_height)
        # total_reward = jump_reward + 2.0 * height_reward
        # return total_reward

        jump_height = self.root_states[:, 2]
        measured_heights = torch.mean(self.rigid_state[:, self.feet_indices, 2]) - 0.06
        jump_diff = jump_height - measured_heights
        # jump_reward = torch.where(jump_diff > 0.5458,  torch.exp(-0.5 * torch.abs(jump_diff)) + 0.5 * jump_diff.clamp(0, 0.5),
        #                           torch.tensor(0.0, device=self.device))
        # return jump_reward
        # feet_contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
        # is_grounded = torch.any(feet_contact, dim=1)
        # jump_reward = torch.where(jump_diff > 0.5458,
        #                           torch.exp(-0.5 * torch.abs(jump_diff)) + 0.5 * jump_diff.clamp(0, 0.5), 0.0)
        #
        # # 增加额外的推力奖励
        # if torch.mean(jump_diff) > 0.5 and is_grounded.all():
        #     self.root_states[:, 9] += 5.0  # 添加一个向上的推力
        #
        # return jump_reward
        jump_reward = torch.exp(-0.5 * torch.abs(jump_diff)) + 0.5 * jump_diff.clamp(0, 0.5)
        return jump_reward

        # measured_heights = torch.mean(self.rigid_state[:, self.feet_indices, 2]) - 0.06
        # jump_diff = self.ref_feet_pos[:, 2] - measured_heights
        # jump_reward =  torch.exp(-0.5 * torch.abs(jump_diff)) + 0.5 * torch.abs(jump_diff).clamp(0, 0.5)
        # jump_reward = torch.exp(-2 * torch.norm(jump_diff)) - 0.3 * torch.norm(jump_diff).clamp(0, 0.3)
        # return jump_reward

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
        knee_left_pos = self.ref_dof_pos[:, 3]  # 左膝关节
        knee_right_pos = self.ref_dof_pos[:, 8]  # 右膝关节
        max_knee_extension = torch.max(knee_left_pos, knee_right_pos)
        return torch.exp(max_knee_extension * 1.0)  # 给予高度奖励

    def _reward_soft_landing(self):
        # 奖励软着陆 - 惩罚较大的着陆冲击力
        contact_force = self.contact_forces[:, self.feet_indices, 2]
        landing_impact = torch.norm(contact_force, dim=1)
        return torch.exp(-landing_impact * 500)  # 给予轻柔着陆奖励

    # def _reward_contact(self):
    #     feet_contact = self.contact_forces[:, self.feet_indices, 2] > 5.0
    #     contact_reward = torch.all(feet_contact, dim=1).float()
    #     return contact_reward

    def _reward_balance(self):
        lin_vel = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return torch.exp(-lin_vel * 0.1)

    def _reward_collision(self):
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                     dim=1)


    
    

# python scripts/train.py --task=x02_ppo_jump --run_name=v1 --headless --num_envs=4096
# python scripts/play.py --task=x02_ppo_jump --run_name v1 --num_envs 64
# python scripts/sim2sim.py --load_model ../logs/x02_ppo_jump/exported/policies/policy_1.pt
