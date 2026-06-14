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

class x02comEnv(LeggedRobot):

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
        self.com_timer = torch.zeros(self.num_envs, device=self.device)
        self.slip_timer = torch.zeros(self.num_envs, device=self.device) # 添加一个跳跃计时器

        self.jump_z = torch.zeros(
            self.num_envs, 1, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.v_z = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)

        self.p_x = self.root_states[:, 0]
        self.v_x = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.jump_zh = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.jump_zk = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)
        self.jump_za = torch.zeros(
            self.num_envs, dtype=torch.float,
            device=self.device, requires_grad=False)

        self.compute_observations()

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
        phase = (self.episode_length_buf) * self.dt
        return phase

    def _get_gait_phase(self):
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # # left foot stance
        stance_mask[:, 0] = self.jump_zk < -0.1
        # # right foot stance
        stance_mask[:, 1] = stance_mask[:, 0]
        # Double support phase
        # stance_mask[sin_pos < 0] = 1
        # stance_mask[self.com_timer == 0] = 0
        # stance_mask[self.com_timer == 1] = 0
        # stance_mask[self._get_phase() % 1.98 > 1.1] = 1

        # stance_mask[self.episode_length_buf > 150] = 1
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
            torch.where(t > 0.4, px2, px1)))) * 0
        
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
    #     t = phase % 1.98
    #     t = self._get_phase()
    #     py = torch.ones_like(phase) * 0.05
    #     px = torch.ones_like(phase) * 0.0 + 0.0 * abs(torch.sin(2 * torch.pi * phase))

    #     px1 = 0.79687 * (t - 0.0) ** 3 - 0.22761 * (t - 0.0) ** 2 + 0.02091 * (t - 0.0) + 0.01962
    #     px2 = -2.99073 * (t - 0.42) ** 3 + 2.83325 * (t - 0.42) ** 2 + 0.21099 * (t - 0.42) + 0.04366
    #     px3 = -9.69247 * (t - 0.61) ** 3 + 8.61694 * (t - 0.61) ** 2 - 2.64865 * (t - 0.61) + 0.12277
    #     px4 = -3.97864 * (t - 1.01) ** 3 - 1.0657 * (t - 1.01) ** 2 + 0.94157 * (t - 1.01) - 0.16949
    #     px5 = 0.12064 * (t - 1.2) ** 3 - 0.2467 * (t - 1.2) ** 2 + 0.18683 * (t - 1.2) - 0.05559
    #     # pz = torch.ones_like(phase) * 0.7825 - 0.2 * abs(torch.sin(2 * torch.pi * phase))
    #     pz = torch.ones_like(phase) * 0.7 - 0.0 * (torch.cos(2 * torch.pi * phase))
    #     pz1 = 11.05639 * (t - 0.01) ** 3 - 6.56588 * (t - 0.01) ** 2 + 0.04302 * (t - 0.01) + 0.6992
    #     pz2 = -29.49624 * (t - 0.42) ** 3 + 14.52348 * (t - 0.42) ** 2 - 0.20425 * (t - 0.42) + 0.37244
    #     pz3 = -12.8121 * (t - 0.61) ** 4 + 11.70907 * (t - 0.61) ** 3 - 3.46796 * (t - 0.61) ** 2 + 0.19291 * (
    #             t - 0.61) + 0.71137
    #     pz4 = 29.29793 * (t - 1.01) ** 3 - 2.14202 * (t - 1.01) ** 2 - 2.0858 * (t - 1.01) + 0.64868
    #     pz5 = 0.4464 * (t - 1.2) ** 4 - 0.16473 * (t - 1.2) ** 3 - 0.89593 * (t - 1.2) ** 2 + 0.99609 * (
    #             t - 1.2) + 0.36358


    #     # while (self.episode_length_buf > 198).any():  # 检查是否有大于 218
    #     #     self.episode_length_buf[self.episode_length_buf > 198] -= 198  # 对所有大于 218 减去 218

    #     pz[t > 0] = pz1[t > 0]
    #     pz[t > 0.42] = pz2[t > 0.42]
    #     pz[t > 0.61] = pz3[t > 0.61]
    #     pz[t > 1.01] = pz4[t > 1.01]
    #     pz[t > 1.2] = pz5[t > 1.2]
    #     pz = pz + 0.09


    #     px[t > 0] = px1[t > 0]
    #     px[t > 0.42] = px2[t > 0.42]
    #     px[t > 0.61] = px3[t > 0.61]
    #     px[t > 1.01] = px4[t > 1.01]
    #     px[t > 1.2] = px5[t > 1.2]
    #     # px[t == 1.7] = 0.0
    #     # px = px - 0.1

    #     [x, self.jst, jh, jk, ja] = self.setSwPt(px, py, pz)


    #     self.jump_zh = jh
    #     self.jump_zk = jk
    #     self.jump_za = -ja
    #     self.ref_dof_pos = torch.zeros_like(self.dof_pos)
    #     # left foot stance phase set to default joint pos
    #     self.ref_dof_pos[:, 1] = 0.0
    #     self.ref_dof_pos[:, 2] = self.jump_zh
    #     self.ref_dof_pos[:, 3] = self.jump_zk
    #     self.ref_dof_pos[:, 4] = self.jump_za
    #     # right foot stance phase set to default joint pos
    #     self.ref_dof_pos[:, 6] = 0.0
    #     self.ref_dof_pos[:, 7] = self.jump_zh
    #     self.ref_dof_pos[:, 8] = self.jump_zk
    #     self.ref_dof_pos[:, 9] = self.jump_za

    #     self.ref_action = 1 * self.ref_dof_pos

    # def compute_ref_vel(self):
    #     phase = self._get_phase()
    #     t = self._get_phase()
    #     vz1 = 1.18999 * t**3 + 30.88172 * t**2 - 12.73049 * t + 0.05268
    #     vz2 =  -250.37119 * (t - 0.41)**3 - 20.73337 * (t - 0.41)**2 + 23.1536 * (t - 0.41) + 0.01772
    #     vz3 =  -9.81 * (t - 0.6) + 1.94436
    #     vz4 =  -206.50273 * (t - 1)**3 + 143.54032 * (t - 1)**2 - 8.8353 * (t - 1) - 1.97791
    #     vz5 =   0.142 + 0.633 * torch.exp(-20.000 * ((t-1.19) - 0.113) ** 2)
    #     # pz = torch.ones_like(phase) * 0.7825 - 0.2 * abs(torch.sin(2 * torch.pi * phase))
    #     vx1 = 5.41682 * t**3 - 0.36654 * t**2 + 0.0312 * t - 0.0005
    #     vx2 = -80.54291 * (t - 0.41)**3 + 12.09768 * (t - 0.41)**2 + 3.90848 * (t - 0.41) + 0.32456
    #     vx3 = torch.ones_like(phase) * 0.946588
    #     vx4 = 76.30915 * (t - 1)**3 - 26.22602 * (t - 1)**2 - 0.33936 * (t - 1) + 0.94734
    #     vx5 = 0.36098 * (t - 1.2) ** 4 - 0.02926 * (t - 1.2) ** 3 - 0.95088 * (t - 1.2) ** 2 + 0.98696 * (
    #                 t - 1.2) + 0.58201

    #     # while (self.episode_length_buf > 198).any():  # 检查是否有大于 218
    #     #     self.episode_length_buf[self.episode_length_buf > 198] -= 198  # 对所有大于 218 减去 218

    #     self.v_z[t > 0] = vz1[t > 0]
    #     self.v_z[t > 0.41] = vz2[t > 0.41]
    #     self.v_z[t > 0.60] = vz3[t > 0.60]
    #     self.v_z[t > 1.00] = vz4[t > 1.00]
    #     self.v_z[t > 1.19] = vz5[t > 1.19]

    #     self.v_x[t > 0] = vx1[t > 0]
    #     self.v_x[t > 0.41] = vx2[t > 0.41]
    #     self.v_x[t > 0.60] = vx3[t > 0.60]
    #     self.v_x[t > 1.00] = vx4[t > 1.00]
    #     self.v_x[t > 1.19] = vx5[t > 1.19]




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
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
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



    def step(self, actions):

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
        # diff_x = self.base_lin_vel[:, 0] - self.v_x
        # self.diffx[:, 0] = diff_x
        # diff_y = self.base_lin_vel[:, 1]
        # self.diffy[:, 0] = diff_y
        # diff_z = self.base_lin_vel[:, 2] - self.v_z
        # self.diffz[:, 0] = diff_z
        # print(self.base_lin_vel)
        # print(self.base_euler_xyz[:, :3])
        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 18.,  # 1
            # self.diffx * self.obs_scales.lin_vel,
            # self.diffy * self.obs_scales.lin_vel,
            # self.diffz * self.obs_scales.lin_vel,
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


    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
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


    def _reward_feet_contact_number(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 0.1
        stance_mask = self._get_gait_phase()
        reward = torch.where(contact == stance_mask, 1, -0.3)
        return torch.mean(reward, dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        rew = torch.sum(torch.square(self.torques), dim=1)
        # rew[torch.sum(self.torques, dim=1) > 1500] = 100000
        return rew

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

    # def _reward_lin_vel_z(self):
    #     rew = torch.relu(torch.clamp(self.base_lin_vel[:, 2], max=1.5))
    #     # rew = torch.relu(torch.clamp(self.base_lin_vel[:, 0], max=1.5))
    #
    #     cycle_time2 = self.cfg.rewards.cycle_time2
    #     cycle_time1 = self.cfg.rewards.cycle_time1
    #     rew[self.episode_length_buf < (cycle_time1 / 2 * 100)] = 0   # 感觉跟不上
    #
    #     rew[self.episode_length_buf > (cycle_time1 / 2 * 100 + cycle_time2 * 100)] = 0
    #
    #     return rew

    def _reward_jump(self):
        """
        Tracks linear velocity commands along the xy axes.
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        x1 = self.root_states[:, 0]
        x2 = self.p_x

        r = torch.where(x1 > x2 , 1, - 0.5)
        self.p_x = self.root_states[:, 0].clone()
        # diff = torch.where(diff>0,1)
        # self.p_x = self.base_pos[:, 0]
        # r= - (lin_vel_error1 + lin_vel_error2)*10
        return r

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """

        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew = torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
        rew[self._reward_joint_pos() < 0.6] = 0
        return rew

    def _reward_collision(self):
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        """
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)
    def _reward_lin_vel_x(self):
        rew = torch.relu(torch.clamp(self.base_lin_vel[:, 0], max=1.5))
        return rew