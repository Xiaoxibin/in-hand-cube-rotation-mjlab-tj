"""TJ Right hand cube rotation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from in_hand_rotation_mjlab.robots import get_tj_right_hand_cfg
from in_hand_rotation_mjlab.robots.tj_hand.tj_right_constants import (
    TJ_HOME_JOINT_POS,
)
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)


# =============================================================================
# TJ specific configuration
# =============================================================================

CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)

GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_31_filtered.npz")

# Robustness knobs (radians). See _apply_tj_robustness_tuning for the rationale.
# 0.02(10-23 轮) -> 0.0(10-27 轮):+/-0.02 rad 的初始扰动把 fingertip_found_max
# 从 0.97 打到 0.79,策略退化成"少接触轻推"或"抓紧推不动",rotate_finite_diff
# 掉到蓝轮的 1/3~1/6。先回到蓝轮的确定初始姿态,把好策略学出来;策略稳定后
# 再按 0 -> 0.005 -> 0.01 -> 0.02 课程式加回(见 _apply_tj_robustness_tuning)。
RESET_JOINT_NOISE_RAD = 0.0
# Proprioception noise: shared config uses 0.01. 0.02(10-23) -> 0.01(10-27):
# 回到共享配置值(= 蓝轮实际值),而不是 0——play 模式关闭 corruption,这里
# 写 0 会和 play/sim2sim 的观测分布产生偏差;等初始扰动课程通过后再放宽。
OBS_JOINT_POS_NOISE_RAD = 0.01
# =============================================================================
# Environment configuration
# =============================================================================

def tj_right_hand_cube_rotate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create TJ Right hand cube rotation environment configuration."""

    return make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_tj_right_hand_cfg,

        # Start from the validated TJ home pose.
        grasp_init_joint_pos=TJ_HOME_JOINT_POS,

        cube_spawn_pos=CUBE_SPAWN_POS,

        # No TJ grasp cache is available yet.
        # Keep this disabled until a TJ-specific grasp cache is generated.
        grasp_cache_file=GRASP_CACHE_FILE, # None / GRASP_CACHE_FILE

        # TJ is a right-hand model.
        # Do not use the Wuji left-hand yaw convention.
        negate_yaw_rate=False,

        play=play,

        # These names should match the TJ MJCF.
        palm_body_name="Palm_R",
        palm_center_geom_expr="Palm_R_collision",

        # TJ fingertip collision geoms:
        #
        #   tip_thumb_collision
        #   tip_index_collision
        #   tip_middle_collision
        #   tip_ring_collision
        #   tip_pinky_collision
        #
        # The expression matches the five TJ fingertips only.
        fingertip_geom_expr=r"tip_(?:thumb|index|middle|ring|pinky)_collision",

        # Do not reuse LEAP-specific actuator dynamics.
        #
        # TJ XML already contains its own position actuators.
        robot_reflected_armature=None,
        robot_effort_limit=None,
        enable_robot_dynamics_dr=False,
    )


def _apply_tj_reward_tuning(cfg: ManagerBasedRlEnvCfg) -> None:
    """TJ-specific reward tuning (applied on top of the shared task config).

    1. rotate_finite_diff:
       - negative_yaw_rate_penalty=2.0: penalize backward yaw 2x harder than
         forward yaw, so "rotate forward / slip back" cycles are strictly
         unprofitable (fast slips cost -0.5 vs forward cap +0.25).
       - tumble_penalty_weight=0.5: charge for cube angular velocity not
         aligned with the cube's own z axis (tumbling about x/y), the failure
         mode that wedges the cube into a jammed pose. A 1 rad/s flip costs
         -0.5/step, outweighing the contact reward (0.15).
       - drift_mode="exp" with drift_decay_k_tilt=8.0: continuous decay of the
         rotation reward with tilt instead of the two-level step gate, which
         had a cliff at 0.35 rad the policy could camp just inside.
    2. torque_penalty: normalize by 21 joints at the +/-1.0 Nm actuator limit,
       so the raw term lives in [0, 1] instead of [0, 21]. Without this the
       penalty outweighs the rotation reward once the policy learns to push.
    3. work_penalty: clip the mechanical-power signal at +/-1.0 W-equivalent
       so rare contact spikes cannot dominate the squared term.
    4. fingertip_contact: DISABLED (removed). The proven LEAP policy was
       trained with no contact reward at all (its sensor was never even
       created), and the unconditional contact income (near the 3-finger cap
       every step) is the main revenue of the "locked grasp + micro-rotation"
       local optimum. It also distorts the grasp style: the policy leans on
       mid-phalanx contact instead of fingertip grips. Fall prevention is
       already covered by object_fallen (-10) and the cube_fell termination.
       The contact metrics (Episode_Metrics/fingertip_*) are kept for
       monitoring.
    """
    cfg.rewards["rotate_finite_diff"].params["negative_yaw_rate_penalty"] = 2.0
    cfg.rewards["rotate_finite_diff"].params["low_rate_boost"] = 2.0 # 1.5(黄色) -> 回到蓝色 2.0
    cfg.rewards["rotate_finite_diff"].params["tumble_penalty_weight"] = 0.5
    cfg.rewards["rotate_finite_diff"].params["tumble_dead_zone"] = 0.15
    cfg.rewards["rotate_finite_diff"].params["drift_mode"] = "exp"
    cfg.rewards["rotate_finite_diff"].params["drift_decay_k_tilt"] = 8.0
    # k_pos 30→50:让旋转收入随位置漂移衰减更快。k_pos=30 时 pos_err 在
    # 1~2cm(口袋区)还留 ~30-50% 收入,槽内蹭转仍是正收益;50 时 1.2cm 处
    # 只剩 ~20%,口袋旋转严格劣于掌心旋转。
    # cfg.rewards["rotate_finite_diff"].params["drift_decay_k_pos"] = 30.0
    cfg.rewards["rotate_finite_diff"].params["drift_decay_k_pos"] = 50.0 # 30.0(黄色) -> 50.0(蓝色)
    cfg.rewards["rotate_finite_diff"].params["drift_tilt_threshold"] = 0.3  # 0.35(黄色) -> 0.3(蓝色)

    cfg.rewards["torque_penalty"].params["max_abs_torque"] = 1.0
    cfg.rewards["torque_penalty"].params["normalize"] = True

    cfg.rewards["work_penalty"].params["max_abs_power"] = 1.0

    # Cumulative-yaw reward: pays a bonus per quarter turn actually achieved,
    # so visible rotation progress (rad, not rad/s) earns its own reward.
    # The rate reward above rewards spinning fast; this one rewards the cube
    # actually turning further than before. Signed integral: backward slips
    # cancel progress. Drift-gated: stops paying if the cube leaves the palm.
    cfg.rewards["cumulative_yaw"] = RewardTermCfg(
        func=hand_cube_mdp.object_cumulative_yaw,
        weight=0.5, # 1.0(黄色) -> 回到蓝色 0.5
        params={
            "object_name": "cube",
            "negate_yaw_rate": False,   # TJ right hand: CCW positive
            "step_angle": 0.7852, #1.5707963,    # pi/4: bonus per quarter turn
            "bonus": 1.0,
            # 0.02 → 0.012:正好卡在 cube_pose_drift_penalty 的 knee(1.2cm)上。
            # 漂过 1.2cm(口袋区)累计 yaw 停止计费,而 knee 以内的正常旋转
            # 抖动不受重罚 → 口袋旋转收入归零、成本照付,严格劣于掌心旋转。
            #
            # gate 取值历史(两轮实测):
            #   0.012(18-52 轮):整项 episode 收入 max=0.002,等于死代码——
            #     真实旋转的 position_error 就悬在 0.009-0.020,几乎每个
            #     episode 一开始就越过 gate。
            #   0.014(11-11 轮):救活了收入(0.002),但 gate 已放宽到 knee
            #     上方,"口袋计费"的防线略有松动;该轮其余修改同时生效,
            #     无法单独归因。
            # 当前回到 0.012(严格回退到 18-52 状态)。若后续轮次该项收入
            # 再次趋零,优先考虑 0.016 而不是 0.014(数据表明真实 pos_err
            # 中心在 1.0-1.2cm,0.014 可能仍在分布内过紧)。
            # "drift_position_threshold": 0.02,
            # "drift_position_threshold": 0.014,
            "drift_position_threshold": 0.016, # 0.02(黄色) -> 0.016(蓝色)
            "drift_tilt_threshold": 0.35,
            # "stall_penalty": 0.05,
            # "stall_steps_threshold": 25,
        },
    )

    # cfg.rewards["fingertip_contact"] = RewardTermCfg(
    #     func=hand_cube_mdp.fingertip_cube_contact_fraction,
    #     weight=0.08,
    #     params={"sensor_name": "fingertip_cube_contact"},
    # )
    if "fingertip_contact" in cfg.rewards:
        del cfg.rewards["fingertip_contact"]

    # Thumb-only contact subsidy(10-20_11-11 轮,已废弃回退)。
    #
    # 10-19_18-52 复盘:拇指完全展开脱离方块 → 握持失去对压 → 打滑卡死。
    # 尝试的接触补贴失败:二元接触信号区分不了"动态对捏"和"静态贴靠",
    # 策略学出拇指常贴(play 实测 10-20_11-11),姿势比 18-52 更差。
    # 结论:拇指的问题不是"缺接触收入",是没有任何项塑造"接触-松开-
    # 再接触"的节奏。两个极端(零补贴→展开 / 二元补贴→贴死)都验证过,
    # 下一步应奖励"占空比节奏"本身(见 thumb_duty_cycle,未实现)。
    #
    # ---- 以下为 10-20_11-11 版本,回退保留 ----
    # 防压死设计:补贴由 drift gate 门控(同 cumulative_yaw 的 in-bounds
    # 测试)。无条件的接触收入会被钻空子——拇指抵住方块时近区漂移成本
    # 只有 0.8/m,1cm 漂移才 0.008/步 < 收入 0.03/步,"抵住换收入"净赚;
    # 门控后方块一漂出 knee(1.2cm)补贴立即停付,压住方块不再有收入。
    # (实测仍被"贴靠但只在方块安静的间隙收钱"的方式钻了空子。)
    #
    # fingertip_indices=(0,) 选取 sensor 的拇指槽(槽序 = 指尖 geom 正则
    # 解析序,已在运行时验证:0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)。
    # cfg.rewards["thumb_contact"] = RewardTermCfg(   # ← 10-20_11-11 版:漂移门控接触补贴,已回退
    #     func=hand_cube_mdp.drift_gated_contact_fraction,
    #     weight=0.03,
    #     params={
    #         "sensor_name": "fingertip_cube_contact",
    #         "max_contacts": 1,
    #         "fingertip_indices": (0,),
    #         # 与 cube_pose_drift_penalty 的 knee 对齐:漂过 knee 补贴停付。
    #         "drift_position_threshold": 0.012,
    #         "drift_tilt_threshold": 0.35,
    #     },
    # )
    # cfg.rewards["thumb_contact"] = RewardTermCfg(   # ← 首版:无条件收入,有被钻空子的风险
    #     func=hand_cube_mdp.fingertip_cube_contact_fraction,
    #     weight=0.03,
    #     params={
    #         "sensor_name": "fingertip_cube_contact",
    #         "max_contacts": 1,
    #         "fingertip_indices": (0,),
    #     },
    # )

    # Thumb duty-cycle band reward (方向 A,10-20 第二轮新增)。
    #
    # 两个失败版本(18-52 零补贴→拇指展开 / 11-11 二元补贴→拇指贴死)是同
    # 一条占空比轴的两个端点。健康的手内旋转是周期性的"捏-松-再捏"循环:
    # 滑窗接触率落在中间带(~0.3-0.8),贴死 ≈1.0,展开 ≈0.0。本项把节奏
    # 本身变成奖励目标:接触率在带内得满奖,两个极端都归零,对称惩罚两种
    # 失败模式。软边界(cosine ramp)保证带外紧邻处有梯度可学。
    #
    # 剂量核算:满额 0.05/步,需覆盖"正常对捏 3-5mm 抖动"的漂移成本
    # (~0.004/步,两区惩罚近区斜率 0.8/m)的 10 倍,但只占 rotate_finite_diff
    # 满收入(~0.44/步)的 ~11%,不足以养活独立的局部最优。
    # drift gate 与 cumulative_yaw 同款:方块漂出界则停付,"压住换收入"不通。
    #
    # band_lo/band_hi(0.2/0.8)是按真 regrasp 占空比的估计值;首轮训练后
    # 看拇指实际接触率分布再校准(若策略贴在带外不进来,放宽带子)。
    # fingertip_indices=(0,):拇指槽,槽序已运行时验证。
    cfg.rewards["thumb_duty_cycle"] = RewardTermCfg(
        func=hand_cube_mdp.thumb_duty_cycle_band,
        weight=0.05,
        params={
            "sensor_name": "fingertip_cube_contact",
            "fingertip_indices": (0,),
            "window_steps": 10,       # 10 步 = 0.5s 的接触率窗口
            "band_lo": 0.2,           # 完全展开(~0.0)落在带下缘外
            "band_hi": 0.8,           # 完全贴死(~1.0)落在带上缘外
            "edge_width": 0.15,       # cosine 软边界宽度
            "drift_position_threshold": 0.012,
            "drift_tilt_threshold": 0.35,
        },
    )

    # Drift-on-ROTATION gate(黑色轮新增)。
    #
    # 问题:原 cube_pose_drift_penalty 惩罚的是"离 reset 位置多远"(累计/状态),
    # 但"漂多少"只取决于结果,取决于"漂多快"。实测(灰色 15799,400 步):
    # 逐帧净 delta 均值 +0.009 rad,滑移比 |net|/path ≈ 0.99(没有来回蹭),
    # 但 19-20s 段回退步占比升到 0.18——说明策略是**一边贴着方块滑一边转**,
    # 计时器不管。原版本默认 1.6cm→指数衰减 50/m:入口(×0.8,20-25%)→ 滑到
    # 1.6cm 处(×1.0,50-57%)→过 gate 后(×0.11 及以下),2.5cm 处收入只剩
    # 2%,与口袋区的 1% 几乎无差。**要的是"漂到一定距离后没有收入",而不是
    # "越远越少"**——转速奖励按 rad/s 瞬时发放,与滑到多快无关。
    #
    # 本项:把"净位置漂移速率"直接作为惩罚(线性 + 死区)。
    #   cost = drift_rate_weight * max(0, |v_cube| - drift_rate_dead_zone)
    # drift_rate_dead_zone=0.02 m/s:正常转动手内抖动(方块无平移运动)不计;
    #   drift_rate_weight=4.0:0.1 m/s 的滑移 = -0.4/步,与满速旋转收入
    #   (~0.44/步)同量级,而 0.2 m/s 滑移 = -0.8/步,严格劣于"老实转"。
    # 与累计距离惩罚互补:前者管"滑得多快",后者管"最终漂到哪"。
    #
    # 10-27 撤回:10-26 两 seed 轮和初始扰动同时加入,rotate_finite_diff 只有
    # 蓝轮的 1/3~1/6,无法归因。先回到纯蓝奖励面做对照;要重新启用时解除注释
    # 并只改这一项。
    # cfg.rewards["cube_drift_rate_penalty"] = RewardTermCfg(
    #     func=hand_cube_mdp.object_linvel_l1,
    #     weight=-4.0,
    #     params={
    #         "object_name": "cube",
    #         "dead_zone": 0.02,
    #     },
    # )


def _apply_tj_robustness_tuning(cfg: ManagerBasedRlEnvCfg) -> None:
    """TJ-specific robustness tuning (applied on top of the shared task config).

    Motivation: the native sim2sim client drives the same checkpoint as a
    different physics backend (mjwarp in training vs the native C MuJoCo
    solver in sim2sim). With identical control inputs the two engines diverge
    by ~0.07 rad of joint angle after a single step, and the current policy
    amplifies that until the cube is thrown out of the palm within ~20 steps.
    Real hardware will present an even larger mismatch (calibration error,
    backdrive, contact compliance), so the policy has to tolerate a joint
    state that does not exactly match what it commanded.

    Two knobs, both deliberately small so the existing behaviour is preserved:

    1. Initial joint perturbation. ``reset_robot_joints.position_range`` was
       (0.0, 0.0) -- every episode started from the exact grasp-cache pose.
       A uniform +/-``RESET_JOINT_NOISE_RAD`` offset teaches the policy to
       recover from a hand that is not where it expects. 0.05 rad (~2.9 deg)
       is roughly the observed engine gap but still well inside the
       grasp-cache stability basin, so it does not destroy the initial grasp.

    2. Larger proprioception noise. The shared config uses +/-0.01 rad.
       +/-0.02 rad matches what real encoders plus a soft PD loop actually
       exhibit through the grasp, and makes the policy rely less on a
       precise reading of its own joints.

    These are NOT a substitute for actuator dynamics randomization: that needs
    kp/kd/armature/frictionloss fitted by sysid first, and randomizing around
    the current hand-written XML estimates would be worse than not doing it.
    """
    joint_noise = RESET_JOINT_NOISE_RAD
    if joint_noise > 0.0:
        cfg.events["reset_robot_joints"].params["position_range"] = (
            -joint_noise,
            joint_noise,
        )

    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(
        n_min=-OBS_JOINT_POS_NOISE_RAD,
        n_max=OBS_JOINT_POS_NOISE_RAD,
    )


def tj_right_hand_cube_rotate_env_cfg_tuned(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """TJ env config with TJ-specific reward tuning applied."""
    cfg = tj_right_hand_cube_rotate_env_cfg(play=play)
    _apply_tj_reward_tuning(cfg)
    if not play:
        # Observation corruption is disabled in play mode, so the noise knob
        # only matters for training; the reset perturbation would also just
        # make play rollouts non-reproducible.
        _apply_tj_robustness_tuning(cfg)
    return cfg
