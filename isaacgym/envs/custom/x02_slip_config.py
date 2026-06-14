from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class x02slipCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 40
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 64
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 10
        num_envs = 4096
        episode_length_s = 1.98 * 3   #24  # episode length in seconds
        use_ref_actions = False

    class safety:
        pos_limit = 1.0
        vel_limit = 1.5  # 适当放宽速度限制以适应跳跃
        torque_limit = 0.9  # 稍微增加扭矩限制，以应对跳跃所需力量

    class asset(LeggedRobotCfg.asset):
        #file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x02/mjcf/X02Lite_v2.xml'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X02LiteV21/X02LiteV21.xml'
        name = "X02Lite"
        foot_name = "ankle_pitch"
        knee_name = "knee"
        terminate_after_contacts_on = ['pelvis']
        penalize_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6  # 降低噪声水平，以避免跳跃时过多干扰 #1.0
        class noise_scales:
            # 噪声尺度
            dof_pos = 0.05  # 自由度位置的噪声尺度
            dof_vel = 1.5  # 自由度速度的噪声尺度
            ang_vel = 0.5  # 角速度的噪声尺度   # 0.5
            lin_vel = 0.1  # 线速度的噪声尺度
            quat = 0.1  # 四元数的噪声尺度
            height_measurements = 0.1  # 高度测量的噪声尺度

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.5458]  # 略低一些的初始位置，以适应跳跃
        # pos = [0.0, 0.0, 0.96]  # 略低一些的初始位置，以适应跳跃
        pos = [0.0, 0.0, 0.98] # 略低一些的初始位置，以适应跳跃
        default_joint_angles = {
            'L_hip_yaw': 0.,
            'L_hip_roll': 0.,
            'L_hip_pitch': 0.,   # 髋关节初始角度，保持与膝关节相协调的值
            'L_knee_pitch': -0.,  # 根据三次多项式调整的膝关节角度
            'L_ankle_pitch': 0.,  # 髋关节和膝关节之间的比例关系
            'R_hip_yaw': 0.,
            'R_hip_roll': 0.,
            'R_hip_pitch': 0.,   # 右腿的髋关节初始角度0.4
            'R_knee_pitch': -0.,  # 右腿的膝关节初始角度-0.8
            'R_ankle_pitch': 0.,  # 右腿踝关节初始角度0.4
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_yaw': 200.0, 'hip_roll': 200.0, 'hip_pitch': 200.0, 'knee': 200.0, 'ankle': 50}
        damping = {'hip_yaw': 10, 'hip_roll': 10, 'hip_pitch': 10, 'knee': 10, 'ankle': 10}
        action_scale = 0.25 
        decimation = 10

    class sim(LeggedRobotCfg.sim):
        dt = 0.001
        substeps = 1
        up_axis = 1
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.02  # 略增接触偏移
            rest_offset = 0.01  # 略增休息偏移
            bounce_threshold_velocity = 0.2
            max_depenetration_velocity = 1.2  # 增大去穿透速度

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_restitution = True
        restitution_range = [0.2, 1.0]
        randomize_base_mass = True
        added_mass_range = [-3., 3.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5  # 增加推力速度
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.01
        rand_interval_s = 10
        randomize_rigids_after_start = True
        randomize_com_displacement = True
        com_displacement_range = [-0.15, 0.15]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.2]
        randomize_motor_offset = True
        motor_offset_range = [-0.05, 0.05]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = True
        Kd_factor_range = [0.5, 1.5]
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = True
        gravity_range = [-0.5, 0.5]
        randomize_lag_timesteps = True
        lag_timesteps = 6

        randomize_base_com = True
        added_com_range = [[-0.15, 0.15],
                           [-0.15, 0.15],
                           [-0.15, 0.15]]

    class commands(LeggedRobotCfg.commands):
        num_commands = 4  # 跳跃只需要高度控制，不再需要多种方向控制
        resampling_time = 8.
        heading_command = False
        class ranges:
            lin_vel_x = [-0.0, 0.0]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
            heading = [-3.14, 3.14]

    class rewards:
        min_dist = 0.2
        max_dist = 0.5
        target_joint_pos_scale = 0.2
        cycle_time = 0.6
        only_positive_rewards = True
        tracking_sigma = 100 #5 #1
        max_contact_force = 500
        class scales:
            joint_pos = 3.0
            orientation = 3.0 #3.0
            feet_contact_number = 5. #5.

            # tracking_ang_vel = 0.5 #0.5
            # tracking_lin_vel = 0.5

            # jump_forward = 5. #往前
            #jump_in_place = 5. #原地
            # fly = 3 #new 
            # track_joint = 0.1
            # sym = 0.1 #同步，不明显可以权重调制5
            # pitch = 10 #2 防止身体后倾，orientation plus

            torques = -1e-4  # 扭矩惩罚比例
            action_smoothness = -0.5  # 平滑性奖励比例

            #base_height = 0.0
            #foot_height = 0.0
            # lin_vel_x = 1
            #lin_vel_z = 0
            
            #joint_vel = 0.0
            #base_vel =  0.0
            #com_height = 0.0
            #jump_lift = 0.0
            #jump_height = 0.
            #knee = 0.0
            #soft_landing = 0.0  # 软着陆奖励比例
            
            
            #balance = 0.0 #new
            #collision = -0.0
            #feet_distance = 0.2
            #knee_distance = 0.2
            #stand_still = 0.0
            # vxvz_tracking = 5.0

    class normalization:
        friction_range = [0.1, 2.5]
        ground_friction_range = [0.1, 2.5]
        restitution_range = [0, 1.0]
        added_mass_range = [-1., 3.]
        com_displacement_range = [-0.1, 0.1]
        motor_strength_range = [0.9, 1.1]
        motor_offset_range = [-0.05, 0.05]
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]
        joint_friction_range = [0.0, 0.7]
        contact_force_range = [0.0, 50.0]
        contact_state_range = [0.0, 1.0]
        body_velocity_range = [-3.0, 3.0]
        foot_height_range = [0.0, 0.3]
        body_height_range = [0.0, 1.0]
        gravity_range = [-1.0, 1.0]
        motion = [-0.01, 0.01]
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.0
            height_measurements = 5.0
        clip_observations = 4.
        clip_actions = 4.


class x02slipCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.#1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60
        max_iterations = 10000  # 适当减少最大迭代次数
        save_interval = 100
        experiment_name = 'x02_ppo_jump'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
