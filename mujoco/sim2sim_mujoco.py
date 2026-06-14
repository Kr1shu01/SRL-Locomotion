import math
import time
from time import sleep

import torch
import mujoco
import mujoco_viewer
import glfw
import numpy as np
from tqdm import tqdm
from Config_jump import Config
from base.SimBase import SimBase
from base.SimBase import NanoSleep


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y])  # , yaw_z


class Sim2Sim(SimBase):
    def __init__(self, _cfg, _policy):
        super().__init__(_cfg, _policy)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        self.default_joint = self.cfg.robot_config.default_joint.copy()
        self.data.qpos[7:17] = self.default_joint[:]
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.q  = np.zeros(10, dtype=np.float32)
        self.dq = np.zeros(10, dtype=np.float32)
        self.euler = np.zeros(2, dtype=np.float32)
        self.gyro  = np.zeros(3, dtype=np.float32)
        self.ref   = np.zeros(10, dtype=np.float32)
        self.jump_timer = 0
        self.kp = self.cfg.robot_config.kpSim.copy()
        self.kd = self.cfg.robot_config.kdSim.copy()

    def asin(self, x):
        min = -1 * np.ones_like(x)
        max = 1 * np.ones_like(x)
        x = self.bnd(x, min, max)
        return np.arcsin(x)

    def bnd(self, x, min_val, max_val):
        return np.maximum(np.minimum(x, max_val), min_val)

    def acos(self, x):
        min = -1 * np.ones_like(x)
        max = 1 * np.ones_like(x)
        x = self.bnd(x, min, max)
        return np.arccos(x)

    def VlegIK(self, vh, vl, va):
        # 从虚拟腿到膝关节腿
        LGS = 0.35
        LGT = 0.35
        t2 = LGS * LGS
        t3 = LGT * LGT
        t4 = vl * vl
        t5 = 1.0 / LGS
        t6 = 1.0 / LGT
        t7 = -np.pi
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
        return [0, jh, jk, ja]

    def setSwPt(self, px, py, pz):
        DTS = 0.066
        DSH = 0.02649
        LSH = 0.08585
        EPS = 1e-6  # 请根据实际情况调整这个值

        pxt = px  # = -0.0
        px -= 0.000
        pyt = py  # = 0.09249
        pzt = pz  # = 0.76585
        pat = pa = 0

        dy = py - DTS
        if np.abs(DSH + dy) < EPS:
            t = -DSH / pz
        else:
            t = -(pz - np.sqrt(-DSH * DSH + dy * dy + pz * pz)) / (DSH + dy)

        js = 2 * np.arctan(t)
        l1 = pz / np.cos(js)
        l2 = l1 + DSH * np.tan(js) - LSH
        ax = self.asin(np.sin(pa) / np.cos(js))  # cos(js) = sin(pa) / sin(ax);
        vl = np.sqrt(px * px + l2 * l2)  # vl = min(vl, cfg['VLM'])
        vh = np.arctan(px / l2)
        va = vh + ax
        jst = js
        return self.VlegIK(vh, vl, va)

    def compute_ref_state(self, phase):
        # Calculate sine of the phase
        phase = np.asarray(phase, dtype=np.float64)
        # phase1 = np.asarray(phase, dtype=np.float64)
        # if phase1 < 1:
        #     phase = 1
        sin_pos = (0.2 * (-np.cos(2 * np.pi * phase))) + 0.3
        py = np.ones_like(phase) * 0.05
        px = np.ones_like(phase) * 0.0 + 0.0 * abs(np.sin(2 * np.pi * phase))
        # pz = np.ones_like(phase) * 0.7825 - 0.2 * abs(np.sin(2 * np.pi * phase))
        pz = np.ones_like(phase) * 0.574 + 0.2 * abs(np.sin(2 * np.pi * phase))
        [x, jh, jk, ja] = self.setSwPt(px, py, pz)

        # Initialize reference degrees of freedom positions
        ref_dof_pos = np.zeros(10)
        # Left foot stance phase set to default joint positions
        # sin_pos_l[sin_pos_l > 0] = 0
        ref_dof_pos[1] = 0.0
        ref_dof_pos[2] = jh
        ref_dof_pos[3] = jk
        ref_dof_pos[4] = -ja
        # print(ja)
        # Right foot stance phase set to default joint positions
        # sin_pos_r[sin_pos_r < 0] = 0
        ref_dof_pos[6] = 0.0
        ref_dof_pos[7] = ref_dof_pos[2]
        ref_dof_pos[8] = ref_dof_pos[3]
        ref_dof_pos[9] = ref_dof_pos[4]

        return ref_dof_pos

    def get_obs(self, cnt_pd_loop):
        q = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        dq = self.data.qvel.astype(np.double)[-self.cfg.env.num_actions:]
        omega = self.data.sensor('gyro').data.astype(np.double)
        quat = self.data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
        euler = quaternion_to_euler_array(quat)
        # add noise
        if self.cfg.noise_scales.add_noise:
            q += (2.*np.random.rand(self.cfg.env.num_actions)-1.) * self.cfg.noise_scales.dof_pos
            dq += (2.*np.random.rand(self.cfg.env.num_actions)-1.) * self.cfg.noise_scales.dof_vel
            euler += (2.*np.random.rand(2)-1.) * self.cfg.noise_scales.euler
            omega += (2.*np.random.rand(3)-1.) * self.cfg.noise_scales.ang_vel
        euler[euler > math.pi] -= 2 * math.pi
        # 滤波
        self.state_filter(q, dq, euler, omega)
        # cal obs

        # phase = (cnt_pd_loop - 400) * 0.001 / self.cfg.control.cycle_time
        #
        # if cnt_pd_loop <= (40) * 10:
        #     phase = 0
        cycle_time = 0.5
        phase = (cnt_pd_loop) * 0.001 / cycle_time
        x = cnt_pd_loop % 250
        if x == 0.00:
            self.jump_timer += 1
        if self.jump_timer >= 2:
            phase = 0

        self.ref = self.compute_ref_state(phase)

        obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
        obs[0, 0] = math.sin(2 * math.pi * phase)  # x * 0.001, ms -> s
        obs[0, 1] = math.cos(2 * math.pi * phase)  # x * 0.001, ms -> s
        obs[0, 2] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = self.cfg.cmd.yaw * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 5:15] = (self.q - self.default_joint) * self.cfg.normalization.obs_scales.dof_pos
        obs[0, 15:25] = self.dq * self.cfg.normalization.obs_scales.dof_vel
        obs[0, 25:35] = self.action
        obs[0, 35:38] = self.gyro * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 38:40] = self.euler
        obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        tauc = self.data.actuator_force
        lf = self.data.sensor('LaF').data.astype(np.double) # foot force
        rf = self.data.sensor('RaF').data.astype(np.double)
        return self.q, self.dq, obs, self.euler, self.gyro, tauc, lf, rf

    def set_sim_target(self, target_q):
        tau = (target_q - self.q) * self.kp - self.dq * self.kd
        self.data.ctrl = np.clip(tau, -self.cfg.robot_config.tau_limit,
                      self.cfg.robot_config.tau_limit)  # Clamp torques
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def run(self):
        self.start_key()
        self.creat_data_file()
        cnt_pd_loop = 0
        pbar = tqdm(range(int(self.cfg.env.run_duration / 0.001)),
                    desc="x02 Simulating...")  # x * 0.001, ms -> s
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            # Obtain an observation
            q, dq, obs, euler, gyro, tauc, lf, rf = self.get_obs(cnt_pd_loop)
            # 1000hz -> 100hz
            if cnt_pd_loop % self.cfg.control.decimation == 0:
                self.target_q = self.get_action(obs) + self.ref  # 策略推理
                # self.target_q = self.ref_trajectory(cnt_pd_loop)  # 参考轨迹可视化，需要将xml中的<freejoint/>注释掉，将机器人挂起来
                # self.target_q = self.gen_traj(cnt_pd_loop)
                now = time.perf_counter()
                pbar.set_postfix(
                    calculateTime=f"{(now - start_time) * 1000:.3f}ms",  # 计算用时，单位毫秒
                    runTime=f"{(now - start):.3f}s"  # 运行时间，单位秒
                )
                # save data
                file_path = self.cfg.save_data.sim_path
                # self.save_data(file_path, cnt_pd_loop/1000., self.target_q, self.data.ctrl, q, dq, tauc, euler,gyro, lf, rf)
            self.set_sim_target(self.target_q)
            cnt_pd_loop += 1

        self.viewer.close()

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        timer = NanoSleep(1)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            for idx in range(self.cfg.env.num_actions):
                self.target_q[idx] = s0 * q0[idx] + s1 * qd[idx]
            self.q = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
            self.dq = self.data.qvel.astype(np.double)[-self.cfg.env.num_actions:]
            self.set_sim_target(self.target_q)
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def init_robot(self):
        self.set_stand_pd()
        final_goal = self.default_joint
        self.joint_plan(1, final_goal)
        for idx in range(self.cfg.env.num_actions):
            self.target_q[idx] = final_goal[idx]
        self.set_walk_pd()

    def show(self):
        m = self.model
        name = m.names.decode('utf-8').split('\x00')
        print("\033[32m>>The robot: %s with %d dof, DofProperties information as follow:\033[0m" % (name[0], m.njnt))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print(
            "\033[33m|     Joint names    | type | idx |   default_pos  | damping | stiffness |   limits_lower   |   limits_upper   | margin |\033[0m")
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        for i in range(0, m.njnt):
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(
                "| %-19s|  %2d  |  %2d | %5.2f(%7.2f) | %5.1f   | %6.1f    | %7.4f(%7.2f) | %7.4f(%7.2f) |  %5.2f |" % (
                    jointName, m.jnt_type[i], i,
                    m.qpos0[i], np.rad2deg(m.qpos0[i]),
                    m.dof_damping[i],
                    m.jnt_stiffness[i],
                    m.jnt_range[i][0], np.rad2deg(m.jnt_range[i][0]),
                    m.jnt_range[i][1], np.rad2deg(m.jnt_range[i][1]),
                    m.jnt_margin[i]))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print("\033[32m>>The robot: %s with %d actuators/controls(ctrl) informations:\033[0m" % (name[0], m.nu))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[33m|     Joint names    | actuator | id | limit | c_lower | c_upper |\033[0m")
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        for i in range(0, m.nu):
            joint_id = m.actuator_trnid[i]  # 获取关节 ID
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id[0])  # 获取关节名称
            actuatorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print("| %-19s|   %-6s | %2d |   %d   | %7.2f | %7.2f |" % (
                jointName, actuatorName, i,
                m.actuator_ctrllimited[i],
                m.actuator_ctrlrange[i][0],
                m.actuator_ctrlrange[i][1]))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[32m>>The robot: %s with %d sensors informations:\033[0m" % (name[0], m.nsensor))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[33m| sensor | id | dim | address |\033[0m")
        print("\033[32m+--------+----+-----+---------+\033[0m")
        for j in range(0, m.nsensor):
            SensorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, j);
            print("| %-6s | %2d |   %d |    %2d   |" % (
                SensorName, j,
                m.sensor_dim[j],
                m.sensor_adr[j]))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[32m>> the end of joint and sensor information !\033[0m")

    def set_stand_pd(self):
        for idx in range(10):
            self.kp[idx] = self.cfg.robot_config.kpStand[idx]
            self.kd[idx] = self.cfg.robot_config.kdStand[idx]
    def set_walk_pd(self):
        for idx in range(10):
            self.kp[idx] = self.cfg.robot_config.kpSim[idx]
            self.kd[idx] = self.cfg.robot_config.kdSim[idx]

    def start_key(self):
        window = glfw.create_window(400, 400, "key_callback", None, None)
        glfw.set_key_callback(window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_W:
            self.cfg.cmd.vx += 0.1
        elif key == glfw.KEY_S:
            self.cfg.cmd.vx -= 0.1
        elif key == glfw.KEY_A:
            self.cfg.cmd.vy += 0.1
        elif key == glfw.KEY_D:
            self.cfg.cmd.vy -= 0.1
        elif key == glfw.KEY_J:
            self.cfg.cmd.yaw += 0.1
        elif key == glfw.KEY_L:
            self.cfg.cmd.yaw -= 0.1
        elif key == glfw.KEY_SPACE:
            self.cfg.cmd.vx = self.cfg.cmd.vy = self.cfg.cmd.yaw = 0
        self.cfg.cmd.vx = np.clip(self.cfg.cmd.vx, -1.0, 1.0)
        self.cfg.cmd.vy = np.clip(self.cfg.cmd.vy, -1.0, 1.0)
        self.cfg.cmd.yaw = np.clip(self.cfg.cmd.yaw, -0.5, 0.5)



if __name__ == '__main__':
    mode_path = "../policies/policy_1-2.pt"
    # mode_path = "policies/policy_1-1.pt"
    policy = torch.jit.load(mode_path)
    mybot = Sim2Sim(Config, policy)
    mybot.show()
    mybot.init_robot()
    mybot.run()