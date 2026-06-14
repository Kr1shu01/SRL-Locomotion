import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import XBotLCfg, x02Cfg, x02slipCfg
import torch
# from humanoid.envs import XBotLCfg



class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

# def compute_ref_state(phase):
#     default_pos = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
#     # Calculate sine of the phase
#     phase = np.asarray(phase, dtype=np.float64)
#     sin_pos = np.sin(2 * np.pi * phase)
#     if sin_pos.ndim == 0:  # If sin_pos is a scalar, make it a 1D array
#         sin_pos = np.array([sin_pos])
#     sin_pos_l = sin_pos.copy()
#     sin_pos_r = sin_pos.copy()
#
#     # Initialize reference degrees of freedom positions
#     ref_dof_pos = np.zeros(10)
#
#     # Scales for the sine modifications
#     scale_1 = 0.2
#     scale_2 = 2 * scale_1
#
#     # Left foot stance phase set to default joint positions
#     sin_pos_l[sin_pos_l > 0] = 0
#     ref_dof_pos[1] = 0.0
#     ref_dof_pos[2] = -sin_pos_l * scale_1 * 1 + default_pos[2]
#     ref_dof_pos[3] = sin_pos_l * scale_2 + default_pos[3]
#     ref_dof_pos[4] = -sin_pos_l * scale_1 + default_pos[4]
#
#     # Right foot stance phase set to default joint positions
#     sin_pos_r[sin_pos_r < 0] = 0
#     ref_dof_pos[6] = 0.0
#     ref_dof_pos[7] = ref_dof_pos[2]
#     ref_dof_pos[8] = ref_dof_pos[3]
#     ref_dof_pos[9] = ref_dof_pos[4]
#
#     # Double support phase
#     ref_dof_pos[np.abs(sin_pos[0]) < 0.1] = default_pos
#     return ref_dof_pos

def bnd(x, min_val, max_val):
        return np.maximum(np.minimum(x, max_val), min_val)

def asin(x):
        min = -1
        max = 1
        x = bnd(x, min, max)
        return np.arcsin(x)

def acos(x):
        min = -1
        max = 1
        x = bnd(x, min, max)
        return np.arccos(x)

def VlegIK(vh, vl, va):
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
        t15 = acos(t13)
        t16 = acos(t14)
        jh = t16 + vh
        jk = t7 + t15
        ja = t7 + t15 + t16 + va
        # print([0, self.jst, jh, jk, ja])
        return [0,jh, jk, ja]

def setSwPt( px, py, pz):
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
        ax = asin(np.sin(pa) / np.cos(js))  # cos(js) = sin(pa) / sin(ax);
        vl = np.sqrt(px * px + l2 * l2)  # vl = min(vl, cfg['VLM'])
        vh = np.arctan(px / l2)
        va = vh + ax
        jst = js
        return VlegIK(vh, vl, va)

def compute_ref_state(phase):
    # default_pos = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
    # Calculate sine of the phase
    # phase = np.asarray(phase, dtype=np.float64)
    ref_dof_pos = np.zeros(10)

    t = phase % 1.98
    py = np.ones_like(t) * 0.05
    px = np.ones_like(t) * 0.0 + 0.0 * np.abs(np.sin(2 * np.pi * t))
    pz = np.ones_like(t) * 0.7 - 0.0 * (np.cos(2 * np.pi * t))

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

    px = np.piecewise(t, 
                      [t <= 0.4, 
                       (t > 0.4) & (t <= 0.6),
                       (t > 0.6) & (t <= 1.0), 
                       (t > 1.0) & (t <= 1.19), 
                       (t > 1.19)], 
                      [px1, px2, px3, px4, px5]) * 0       
    pz = np.piecewise(t, 
                      [t <= 0.4, 
                       (t > 0.4) & (t <= 0.6),
                       (t > 0.6) & (t <= 1.0), 
                       (t > 1.0) & (t <= 1.19), 
                       (t > 1.19)], 
                      [pz1, pz2, pz3, pz4, pz5]) + 0.12
    
    [x, jh, jk, ja] = setSwPt(px, py, pz)

    ref_dof_pos[1] = 0.
    ref_dof_pos[2] = jh
    ref_dof_pos[3] = jk
    ref_dof_pos[4] = -ja

    ref_dof_pos[6] = 0.
    ref_dof_pos[7] = ref_dof_pos[2]
    ref_dof_pos[8] = ref_dof_pos[3]
    ref_dof_pos[9] = ref_dof_pos[4]

    return ref_dof_pos

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
    return np.array([roll_x, pitch_y])


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    quat = data.sensor('bq').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd, ref):
    '''Calculates torques from position commands
    '''
    s = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
    return (target_q + ref - q ) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    # default_pos = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
    # data.qpos[7:17] = [0, 0, 0.4, -0.8, 0.4, 0, 0, 0.4, -0.8, 0.4]
    default_pos = [0, 0, 0., -0., 0., 0, 0, 0., -0., 0.]
    data.qpos[7:17] = [0, 0, 0., -0., 0., 0, 0, 0., -0., 0.]
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    # target_q = default_pos
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

    count_lowlevel = 0
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        force = [0, 0, 0]
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # if count_lowlevel >= 600:
            #     phase = 0
            # else:
            phase = count_lowlevel * cfg.sim_config.dt

            ref = compute_ref_state(phase)

            obs[0, 0] = math.sin(2 * math.pi * phase)
            obs[0, 1] = math.cos(2 * math.pi * phase)
            obs[0, 2] = cmd.vx
            obs[0, 3] = cmd.vy
            obs[0, 4] = cmd.dyaw
            obs[0, 5:15] = (q - default_pos)
            obs[0, 15:25] = dq * 0.05 # dq * 0.05
            obs[0, 25:35] = action
            obs[0, 35:38] = omega * 1
            obs[0, 38:40] = eu_ang

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
            # print(data.qvel[2])
            hist_obs.append(obs)
            hist_obs.popleft()

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            for i in range(cfg.env.frame_stack):
                policy_input[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            target_q = action * cfg.control.action_scale * 1.8

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds,ref)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          name="base_link")

        data.xfrc_applied[robot_body_id, :3] += force
        mujoco.mj_step(model, data)
        # print('11111',robot_body_id)

        # viewer.add_marker(pos=data.site_xpos[robot_body_id - 1], label="Applied Force", size=0.1)

        # mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg(x02slipCfg):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/X02LiteV21/X02LiteV21.xml'
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x02/mjcf/X02Lite_v2.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/X02LiteV21/X02LiteV21.xml'
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/x02/mjcf/X02Lite_v2.xml'
            sim_duration = 60.0
            dt = 0.001
            decimation = 10

        class robot_config:
            kps = 1 * np.array([200, 200, 200, 200, 50, 200, 200, 200, 200, 50], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.double)
            tau_limit = 50000. * np.ones(10, dtype=np.double)


    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
