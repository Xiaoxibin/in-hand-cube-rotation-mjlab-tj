1、训练
	注意将 vim .venv/lib/python3.13/site-packages/rsl_rl/utils/wandb_utils.py  原来的
	settings=wandb.Settings(start_method="thread"), 修改 为settings=wandb.Settings(), 然后再进行训练
	$ uv run python scripts/train.py Mjlab-Leap-Left-HandCube-Rotate --env.scene.num-envs 128
	训练结果保存在 wandb 中
2、验证
	预训练好的模型：uv run python scripts/play.py Mjlab-Leap-Left-Custom-HandCube-Rotate --checkpoint-file ckpts/
	leap_left_custom_model_4900.pt
	自己模型：uv run python scripts/play.py Mjlab-Leap-Left-HandCube-Rotate --checkpoint-file logs/rsl_rl/ 
	leap_left_hand_cube_rotate/2026-09-20_17-57-19/model_4300.pt
3、git 相关
	cd ~/Projects/my-project
	git init
	git add .
	git commit -m "Initial commit"

	git remote add origin git@github.com:Xiaoxibin/my-project.git
	git branch -M main
	git push -u origin main
	修改后：
	git add .
	git commit -m "xxx"
	git push
	
4、wuji_left_hand 1 迁移相关
4.1 mjcf 相关 
	下载https://github.com/wuji-technology/wuji-description/releases?page=2 里的 v2026.05.19源文件
	$ mkdir -p src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/assets
	$ touch src/in_hand_rotation_mjlab/robots/wuji_hand/__init__.py
	$ cp ~/下载/wuji-description-2026.05.19/hand/body/mjcf/left.xml src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/left_hand.xml
	$ cp ~/下载/wuji-description-2026.05.19/hand/body/meshes/left/*.STL \
   src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/assets/
   	left_hand.xml 修改路径为 meshdir="./assets/"
	$ uv run python -m mujoco.viewer   --mjcf=src/in_hand_rotation_mjlab/robots/wuji_hand/xmls/left_hand.xml
	
	uv run python -m mujoco.viewer   --mjcf=src/in_hand_rotation_mjlab/robots/tj_hand/xmls/right.xml
	

4.2 环境验证
	uv run python scripts/play.py Mjlab-Wuji-Left-HandCube-Rotate --num-envs 1 --agent zero
	
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --num-envs 1 --agent zero

4.3 grasp_cache文件生成
	代码实现位于scripts/collect_hand_cube_grasp_cache_wuji.py：
	uv run python scripts/collect_hand_cube_grasp_cache_wuji.py   Mjlab-Wuji-Left-HandCube-Rotate
	生成质量与 WUJI_HOME_POS、 CUBE_SPAWN_POS、 cube_size 有关
	产出 cube_size 、joint_pos、 cube_pose_rel
	
	uv run python scripts/collect_hand_cube_grasp_cache_tj.py   Mjlab-TJ-Right-HandCube-Rotate
	
4.4 wuji训练
	uv run python scripts/train.py Mjlab-Wuji-Left-HandCube-Rotate --env.scene.num-envs 128
	uv run python scripts/train.py Mjlab-TJ-Right-HandCube-Rotate --env.scene.num-envs 2048
	
	4.4.1 中断后恢复训练
	uv run python scripts/train.py Mjlab-TJ-Right-HandCube-Rotate --env.scene.num-envs 2048 --agent.resume True --agent.load-run 2026-10-14_11-51-31 --agent.load-checkpoint model_1600.pt

	
4.5 wuji验证 
	uv run python scripts/play.py Mjlab-Wuji-Left-HandCube-Rotate --checkpoint-file logs/rsl_rl/wuji_left_hand_cube_rotate/2026-09-28_18-30-33/model_4999.pt
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt


LEAP Hand ： 538AE5 蓝色

tj 参数：
=============================================================================================================================
绿色：2026-10-09_19-27-11
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt

TJ_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.08)
TJ_HOME_ROT: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0)
TJ_HOME_JOINT_POS: dict[str, float] = {
    # =========================
    # Thumb
    # =========================
    "CMC1": -0.9,
    "CMC2":  0.107,
    "CMC3": 0.26,
    "MCPT": -1.12,
    "DIPT": -0.752,

    # =========================
    # Index
    # =========================
    "MCPI2":  0.05,
    "MCPI1": -0.8,
    "PIPI":  -1.25,
    "DIPI":  -0.6,

    # =========================
    # Middle
    # =========================
    "MCPM2": 0.0616,
    "MCPM1": -0.509,
    "PIPM":  -1.34,
    "DIPM":  -1.07,

    # =========================
    # Ring
    # =========================
    "MCPR2": -0.0621,
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # =========================
    # Pinky
    # =========================
    "MCPS2": -0.27,
    "MCPS1": -1,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}

cube_size : 0.0272
CUBE_SPAWN_POS = (-0.006, 0.043, 0.08)

=============================================================================================================================
棕色 ：2026-10-10_14-12-59
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-10_14-12-59/model_

grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz"
cube_size : 0.022
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -0.9,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -1.45,     # 原 -1.12,更大对掌
    "DIPT":  -0.45,     # 原 -0.752,拇指伸直一点抬高指尖

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,     # 原 0.05
    "MCPI1": -0.95,     # 原 -0.8
    "PIPI":  -1.40,     # 原 -1.25
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,     # 原 0.0616
    "MCPM1": -0.60,     # 原 -0.509
    "PIPM":  -1.40,     # 原 -1.34
    "DIPM":  -1.07,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,     # 原 -0.0621
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.35,
    "MCPS1": -1.00,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}

=============================================================================================================================
灰色（有中断）：2026-10-12_16-51-38 / 2026-10-12_18-37-24
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-12_18-37-24/model_700.pt 

GRASP_CACHE_FILE = (
    "tasks/hand_cube/cache/tj_right_grasp_cache_24.npz"
)
cube_size : 0.024  (0.0225~0.0255)
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -0.9,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -1.45,     # 原 -1.12,更大对掌
    "DIPT":  -0.45,     # 原 -0.752,拇指伸直一点抬高指尖

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,     # 原 0.05
    "MCPI1": -0.95,     # 原 -0.8
    "PIPI":  -1.40,     # 原 -1.25
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,     # 原 0.0616
    "MCPM1": -0.60,     # 原 -0.509
    "PIPM":  -1.40,     # 原 -1.34
    "DIPM":  -1.07,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,     # 原 -0.0621
    "MCPR1": -0.51,
    "PIPR":  -1.35,
    "DIPR":  -0.811,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.25,
    "MCPS1": -1.00,
    "PIPS":  -1.22,
    "DIPS":  -0.613,
}
=============================================================================================================================
红色：2026-10-13_14-31-44 / 2026-10-13_18-34-48
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-13_14-31-44/model_900.pt

GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_24_v1.npz")
1、修改了奖励函数：添加了 tumble_penalty 和 cube_pose_drift_penalty； 2、impratio 改成 100； 3、 rotate_finite_diff 里的 drift_mode 改成了 "exp"；

=============================================================================================================================
紫色：  2026-10-14_11-51-31 / 2026-10-14_16-36-53
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-14_16-36-53/model_1600.pt

1、cube_size 为 0.028  CUBE_SPAWN_POS = (-0.006, 0.0368, 0.08)  GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_28.npz")；
2、env_cfgs.py 中增加 cumulative_yaw 奖励，step_angle 为 1.5707963（pi/2）；

cube_size = 0.028
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -1.45,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -0.7,
    "DIPT":  -0.65,

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,
    "MCPI1": -0.95,
    "PIPI":  -1.15,
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,
    "MCPM1": -0.80,
    "PIPM":  -1.0,
    "DIPM":  -0.6,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,
    "MCPR1": -0.81,
    "PIPR":  -1.05,
    "DIPR":  -0.611,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.15,
    "MCPS1": -1.30,
    "PIPS":  -0.82,
    "DIPS":  -0.611,
}

=============================================================================================================================
青色 ： 2026-10-14_20-33-08

修改：
1、pose_diff_penalty 中的 joint_tolerance 改成了 0.15  # 原 0.4； 2、MCT_CMC3_R 中的 collision type 换成了 box（未改！！！）；

cube_size = 0.031	CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)	 GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_31.npz")
TJ_HOME_JOINT_POS: dict[str, float] = {
    # ===== 拇指:抬起来、从 +x 侧伸向指间,与食/中指形成对捏 =====
    "CMC1":  -1.45,
    "CMC2":   0.107,
    "CMC3":   0.15,
    "MCPT":  -0.7,
    "DIPT":  -0.65,

    # ===== 食指:外展(MCPI2 0.05→0.25)并卷起参与捏握 =====
    "MCPI2":  0.25,
    "MCPI1": -1.1,
    "PIPI":  -0.9,
    "DIPI":  -0.60,

    # ===== 中指:外展并卷起,构成捏握的上/后支撑面 =====
    "MCPM2":  0.25,
    "MCPM1": -0.80,
    "PIPM":  -1.0,
    "DIPM":  -0.6,

    # ===== 无名指:微调位置辅助支撑 =====
    "MCPR2": -0.05,
    "MCPR1": -0.81,
    "PIPR":  -0.7,
    "DIPR":  -0.611,

    # ===== 小指:内收靠拢(MCPS2 -0.27→-0.35) =====
    "MCPS2": -0.15,
    "MCPS1": -1.30,
    "PIPS":  -0.82,
    "DIPS":  -0.611,
}
（漂到“口袋区”后掉落）
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-14_20-33-08/model_4400.pt  / 3500.pt
 


黄色 ： 2026-10-15_13-45-37
	修改 ： 
	1、cube_pose_drift_penalty 权重改为 -1.5 # 原-0.8； position_threshold 改为 0.005 # 原 0.002；
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-15_13-45-37


淡蓝色： 2026-10-15_19-09-05
	修改：
	1、cube_pose_drift_penalty 双区域设计，小幅抖动不惩罚太重 ，超过拐点位置（0.012 m）则大幅漂移惩罚加重；2、奖励中增加 if position_knee is not None ； 
	3、修改了 collect_hand_cube_grasp_cache_tj.py ，加入 max_settle_tilt_deg
	现象：
	立方体被压住不动，且大拇指将立方体给抵出去了
	

深绿色： 2026-10-16_10-57-22
（小拇指和大拇指之间卡住）
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-16_10-57-22/model_6500.pt 
	修改：
	cfg.rewards["rotate_finite_diff"].params["drift_tilt_threshold"] = 0.3  # 原 0.15
	
	
肉色 ： 2026-10-16_17-31-42
(失败,翘翻):pose_diff 提压+stall+MCPS1-1.0+新cache,从头训 → 教训:这些修改不能叠加在从头训上;run 下一轮:C 基线 + 仅 pose_diff 加压,resume → 观察堵塞是否付费消失
	修改：
	1、pose_diff_penalty 惩罚项(当前 21 个关节角相对 episode(home pose)的偏差)，weight=-0.2, # 原 -0.1 ,  "joint_tolerance": 0.1,
	2、pose_diff_penalty 的课程，"weight_min": -0.05, # 原 -0.01  "weight_max": -0.2, # 原 -0.1
	3、cumulative_yaw 停滞惩罚：
		rewards.py 中的新增；
		env_cfgs.py中 params 显式声明
	4、 初始关节 "MCPS1": -1.0,  # 原 -1.30；  CUBE_SPAWN_POS = (-0.00, 0.032, 0.08) # 原 (-0.006, 0.032, 0.08)；
		GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_31_MCPS1.npz")

青绿色 ： 2026-10-19_11-47-34
（，卡住）
	修改：
	pose_diff_penalty 加压(weight −0.2 + 课程 −0.02/−0.2)
	
	
	
褐色 ： 2026-10-19_18-52-49
（旋转后 大拇指展平，卡死终止 reset）
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-19_18-52-49/model_10800.pt
	修改：
	① pose_diff_penalty 弱化 — hand_cube_env_cfg.py:359-375
	weight −0.2 → −0.02(象征性正则，原值留注释)
	tolerance 保持你已改回的 0.4,注释记录了 0.15 的历史
	
	② 反 curriculum 修复 — hand_cube_env_cfg.py:520-577
	四个递增课程(linvel / pose_diff / torque / work)全部注释掉，curriculum 字典现在为空
	权重固定为课程区间的温和中间值：
	object_linvel_penalty: −0.3 → −0.1
	torque_penalty: −0.1 → −0.3(注意 TJ 层开了 normalize,区间 [0,1])
	work_penalty: −0.05 → −0.03
	
	③ 口袋区停止计费 — env_cfgs.py
	rotate_finite_diff 的 drift_decay_k_pos: 30 → 50(30 留注释，口袋区收入衰减更快)
	cumulative_yaw 的 drift_position_threshold: 0.02 → 0.012(“正好卡在 knee 上”)
	cube_pose_drift_penalty 的两区设计(knee 1.2cm / far_weight 3.0)保持不动——它负责“口袋贵”，上面两项负责“口袋没收入”
	
	④ stall termination(新增) — terminations.py:29-116 新增 object_yaw_stalled 类，hand_cube_env_cfg.py 注册 cube_stalled 终止项：
	滑动窗口 50 步(2.5s)内去抖后的净 yaw 转动 < 0.05 rad → 提前终止
	用带死区的累计净转动而不是瞬时角速度，慢速真旋转和 rotate-forward/slip-back 周期处理都已考虑
	你在 rewards.py 里注释掉的 stall_penalty 方案就正式废弃了(termination 形式信号更强)
	⑤ entropy_coef — rl_cfg.py:32-36:0.003 → 0.008(原值留注释)

水粉 ： 2026-10-20_11-11-15
	修改：
	uv run python scripts/train.py Mjlab-TJ-Right-HandCube-Rotate \
  --env.scene.num-envs 2048 --agent.resume True \
  --agent.load-run 2026-10-19_18-52-49 --agent.load_checkpoint model_10800.pt \
  --agent.seed 1

	seed = 1
	entropy_coef 回落 — rl_cfg.py:32-38		entropy_coef=0.003, # 原 0.008
	cumulative_yaw gate 救活 — env_cfgs.py:145	"drift_position_threshold": 0.014 # 原 0.012
	拇指接触补贴(新增)— 两处

灰色 ： 2026-10-20_18-17-22
(成功，后期reset终止)
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-20_18-17-22/model_15799.pt \
  --num-envs 4

	修改：
	entropy_coef=0.003
	drift_position_threshold: 0.016
	新增 thumb_duty_cycle(weight 0.05, window 10 步, band 0.2~0.8, edge 0.15, drift gate 0.012)
	seed 1
	
蓝色 ： 2026-10-21_15-58-28
（旋转后期  reset 终止，和灰色相比几乎没增长）
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-21_15-58-28/model_19700.pt --num-envs 1
	修改 ：rl_cfg.py超参数
	entropy_coef=0.0005,      # was 0.003
	learning_rate=3e-4,       # was 1e-3
	schedule="adaptive",      # 保持 adaptive
	desired_kl=0.02,          # was 0.01

play 视频录制：
uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate   --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-21_15-58-28/model_19700.pt   --num-envs 1 --video True   --video-length 1200 --video-fps 30   --video-height 720 --video-width 1280
sim2sim 视频录制：
uv run python scripts/sim2sim.py Mjlab-TJ-Right-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-21_15-58-28/model_19700.pt


黄色 ： 2026-10-22_16-55-22
（失败，回退到蓝色曲线）
	修改：
	1、提高转速上限激励，让 “更快”有收益
	cfg.rewards["rotate_finite_diff"].params["clip_max"] = 0.5     # 0.35 -> 0.5, 需 0.25 rad/s 才满额
	cfg.rewards["rotate_finite_diff"].params["low_rate_boost"] = 1.5  # 2.0 -> 1.5, 慢转的优惠减少(前期已渡过)
	2、里程奖励提权，让“转了多少”比“转得多块”更重要
	cfg.rewards["cumulative_yaw"].weight = 1.0     # 0.5 -> 1.0

	entropy_coef=0.001, #0.005/0.003
        learning_rate=5e-4, # 3e-4 / 1e-3


	新修改：
	RESET_JOINT_NOISE_RAD = 0.02
	OBS_JOINT_POS_NOISE_RAD = 0.02
	entropy_coef = 0.003


=============================================================================================================================
水粉 ： 2026-10-15_11-14-03

	修改：
	1、cube_pose_drift_penalty 权重改为 -1.5 # 原-0.8； position_threshold 改为 0.005 # 原 0.002；
	cube_size = 0.033	CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)	 GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_33.npz")
	现象：
	立方体太大，会掉落；cube_pose_drift_penalty 权重太大，立方体不转
	uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-15_11-14-03

我该如何和leader解释play 和 sim2sim 不一致的原因，如果 sim2sim 效果不好，是不是代表 sim2real效果也不好？我现在还没搞懂sim2sim存在的原因,和sim2real的关系是什么？

仿真：

网线直连控制器,主机 IP 与 6.6.7.190 同网段(ifconfig 确认,ping 6.6.7.190)

标定：
raw_int16 = (JOINT_SIGN * q_sim + JOINT_ZERO_OFFSET_RAD) * RAD2HW     # 下发给硬件
q_sim     = JOINT_SIGN * (raw_int16 * HW2RAD - JOINT_ZERO_OFFSET_RAD) # 硬件读回

实验 0:连接与只读检查
cd real_world
python tj2_interface.py --read

实验 1:方向标定(核心)
python tj2_interface.py --calibrate --joints 0 1 2 3 4     # 先只标拇指
# 或全部 21 个
python tj2_interface.py --calibrate

实验 2:零位标定
# 1) 断电/失能，把真机摆到 MJCF 定义的全零姿态（q_sim = 0）
# 2) 读取原始值
python tj2_interface.py --read
# q_hw = sign*q_sim + offset,在仿真零位 q_sim=0 时 offset = raw*HW2RAD

实验 3:限位验证
对每个关节慢慢指令到 JOINT_LIMITS_RAD 的上下限，确认：机械上确实能到、不顶死、不异响、不堵转。表里的值来自 XML 的 ctrlrange

实验 4:正弦冒烟 + 姿态对照
python tj2_interface.py --sine --fingers thumb --seconds 6



tj2_sysid_collect.py（实物机上，我刚写的）      sysid_fit_from_log.py（GPU 机器上，已有的）
─────────────────────────────────         ─────────────────────────────
作用：让真手做激励动作（阶梯/啁啾/PRBS），      作用：拿采集的 NPZ 数据，在 GPU 上并行仿真
     同步记录「下发的命令 vs 实际关节角」           8192 个副本，每个副本用一组不同的执行器
     存成 NPZ                                    参数回放同样的命令序列，比较哪组参数的
                                                仿真轨迹和真实轨迹最接近（RMSE 最小）
输出：data/tj2-sysid/index_XXXX.npz           输出：*_sysid_fit.npz —— 每关节 5 个参数的拟合结果

1、让手指做慢速轨迹，记录 cmd vs 实际 pos，用 scripts/sysid_fit_from_log.py 拟合 TJ2 的增益/摩擦，更新回 sim 的执行器参数

# 1) 采集(每根手指一个文件，拇指优先)
python real_world/tj2_sysid_collect.py --fingers thumb --chirp-s 30 --prbs-s 60
python real_world/tj2_sysid_collect.py --fingers index middle ring pinky

# 2) 拟合(需要 CUDA)
uv run python scripts/sysid_fit_from_log.py Mjlab-TJ-Right-HandCube-Rotate \
  --log-file data/tj2-sysid/thumb_<ts>.npz --finger thumb --optimizer cem

# 3) 检查 per_joint_rmse_eval，确认拟合质量后再更新 XML 标称值
# 4) 填 robot_reflected_armature / robot_effort_limit，再打开执行器 DR（窄范围起步）


# 进程内模式（一个终端）
uv run python scripts/sim2sim.py Mjlab-TJ-Right-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<TJ实验>/<时间戳>/model_XXXX.pt \
  --num-steps 2000

# 服务器模式（终端1）
uv run python scripts/policy_server_zmq.py Mjlab-TJ-Right-HandCube-Rotate \
  --checkpoint-file logs/rsl_rl/<TJ实验>/<时间戳>/model_XXXX.pt \
  --bind tcp://0.0.0.0:5555
# 服务器模式（终端2）
uv run python scripts/sim2sim.py Mjlab-TJ-Right-HandCube-Rotate \
  --server-endpoint tcp://localhost:5555 \
  --num-steps 2000






待完成：
已知指尖相对于手腕的位姿，利用逆运动学求解手指的各关节角度。
1、https://github.com/DexRobot/dexrobot_kinematics
2、https://github.com/dexsuite/dex-retargeting
3、https://github.com/mimicrobotics/mimic_retargeter_lab
4、


需要更正下你，LEAP Hand 的 link 和 palm 的实物近似为 box ,因为它的实物 本身就是个立方体，所以 MJCF的collision 为box ,但是 wuji 和 tj hand的link 不是立方体，所以 用 box 来建立 collision 几何是否有问题？
另外，原来作者只有 tj hand 的 URDF 文件，这份 MJCF 文件是我写的（可能很多地方都有问题），不具有参考价值。
帮我检查下，然后生成新的  tj hand 的 MJCF 文件。
为什么我已经有各个link的 .STL 文件，确不能用mesh来做碰撞几何，为什么要用 box？用 box 会不会和实物手不匹配？
告诉我  4 处 Python 正则 在哪修改，如何修改？
我的collect_hand_cube_grasp_cache_tj.py 文件似乎有些问题，它只会保存更新 cube 的位姿和大小，但是不会更新手的 joint pose ,这样对吗？如果不对的话该如何修改？
另外，xml中的 armature 以及  tj hand 的 rl_cfg.py 参数该如何设定才能让训练效果更好？其他还需要注意哪里，才会让整个任务完成度更高，效果更好？
再检查下我的奖励函数的各个项，以及各个项之间的系数设定合理吗？
我的 TJ_HOME_POS: tuple[float, float, float] = (0.0, 0.0, 0.08)，CUBE_SPAWN_POS = (-0.006, 0.043, 0.08)，cube_size 为 0.0272(半边长)，为什么不会导致立方体穿透灵巧手的手掌面？TJ_HOME_ROT: tuple[float, float, float, float] = (0.707107, -0.707107, 0.0, 0.0)灵巧手先绕x轴旋转90度后掌面朝上，(0.0, 0.0, 0.08) 应该是腕部的位置，那掌面的高度应该也是 0.08,cube的中心高度也为0.08,这样不会导致立方体穿过手掌吗，可以是我可视化 uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --num-envs 1 --agent zero 却没有穿过，立方体确实落在手掌上，这是什么原因？
我的电脑能跑动 env.scene.num-envs 4096 吗？
你前面提到的奖励函数的两个问题和缺失项我该如何修改？
那我手掌的实际高度在哪呢？
我可以直接拿 scripts/collect_hand_cube_grasp_cache_tj.py 来生成 tj hand 的cache 文件吗？还需要修改下其他地方了吗？
我已经通过 uv run python scripts/collect_hand_cube_grasp_cache_tj.py   Mjlab-TJ-Right-HandCube-Rotate 生成了 cache 文件，接下来需要训练了，在训练前检查下 tj hand cube rotate 这个项目代码是否有不合理的地方，如何训练？


把 reward 里负 yaw_rate 乘个 1.5~2 倍系数，或把 clip_min 收窄到 −0.1——让“打滑倒退”变得比“不转”更亏。 这个代码在哪修改？如何修改?ingertip_contact 权重调高在哪调？如何调？
帮我把  奖励不对称修正 写进代码
如果我在cube_size=0.272,grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz" 下训练了一个策略，后面我将cube_size 改为了0.024，换了个初始灵巧手姿态，grasp_cache_file = None再来运行这个策略：uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt，会出现什么样的结果？能这样操作吗？参数已经训练好了换一个立方体和初始姿态可以吗？
改动了cube_size=0.022 和 奖励不对称修正 后重新训练了，运行  uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/ 查看到立方体转动后会朝着大拇指、食指和中指那块，然后被那几根手指卡住动不了，立方体朝着食指那方向的掌面，其他掌面地方没有和立方体接触了，而且几根手指都太卷曲了，没有伸展开。检查下什么原因？为什么LEAP hand 的cube rotate 任务效果可以这么好？参考下 LEAP Hand 的一些参数：https://github.com/Msornerrrr/in-hand-rotation-mjlab/tree/main 我需要调整哪些参数？
如果我拿 grasp_cache_file = "tasks/hand_cube/cache/tj_right_grasp_cache.npz" 去训练，由于立方体的尺寸是一个范围，如果最终的效果比较好的 checkpoint 文件是 model_1200.pt，那我怎么知道这个文件取的 cube 的尺寸是多少？
Curriculum 在训练过程中参数也会更新吗？ Curriculum 是什么？和损失函数之间有什么区别？
CUBE_SPAWN_POS 和 TJ_HOME_POS 位置没搞懂
rotate_finite_diff 到底如何计算的？它的单位是什么？是 rad/s 还是 rad？这个 hand cube rotate 任务的评价指标似乎不涉及立方体到底转了多少，是不是持续地在旋转，而是旋转的速度，是这样的吗？我希望真真切切看到立方体转起来了，不管速度快慢，至少在 圈数的弧度上得看到旋转，旋转的圈数得增加，为什么之前那个奖励和 curriculum 对 LEAP Hand 效果这么好，到我们这效果却那么差到底差在哪了？是手的结构？手的 home joint pose？立方体的尺寸？还是奖励函数或者 curriculm 或者 域随机化 DR 的差别？

如果我 2026-10-09_19-27-11 的训练没有 cube_pose_drift_penalty 这项奖励，并且也没有 cube_pose_deviation 这个终止条件，但是有 grasp_cache_file 这个 cache 文件，可能还有些其他的训练参数修改了，现在我能 uv run python scripts/play.py Mjlab-TJ-Right-HandCube-Rotate --checkpoint-file logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-09_19-27-11/model_1200.pt 来查看效果吗？

训练过程中我需要学习下这个项目，从这个项目涉及到的强化学习基本知识，再到这个项目中一些算法知识，再到这个项目中的一些部署和优化知识，给我介绍一下。
接下来如果我想 sim-to-real 实物上，该从哪里入手，需要了解哪里的代码？实物部署 tj hand 的话是参考代码中的 real_world/README.md 还是参考PIPELINE.md?这两个有什么区别？

2000轮效果不错，接下来是不是该50000个 iteration 继续训练或者增大哪个训练参数让效果更好 ？
需要添加噪声或者域随机化的注释给放开？
当前有加什么域随机化或者噪声吗？

如果到 2000 iter rot_prog 还停在 0.08 以下，那时的问题就不是“会不会转”而是“转不快”，可能的方向是提高 target_yaw_rate 的梯度效率(比如把 clip_max 提到 0.3)或检查 target 0.20 rad/s 对 TJ 手是否过于激进——但那是 1500 iter 之后的事，现在让子弹飞。

learning_time很低，一般都是在0.028附近振荡，collection_time 很高，在 10 附近振荡，mean_reward 最后在 4 附近振荡


撤 boost(rewards.py 参数回 1.0 或删)，保留 clip_max 0.35~0.5;
课程棘轮化(curriculums.py);
drift 死区 0.005→0.003;

可以考虑下 考虑换主奖励形式：用“累计正确旋转角度”(episode 内 CCW 累计弧度，倒转扣分)替代“瞬时角速度”——它天然不可钻(想拿分必须真转)，代价是需要防“甩 cube 刷角度”(用现有的 drift gate + linvel 罚压制)。这个怎么改

手内旋转任务对立方体更简单还是对圆柱体更简单？


待完成：
先 retargetting得到像人的动作，再由强化或模仿学习符合规范的动作
把模仿学习（IL）作为基础或引导，再通过强化学习（RL）进行微调与策略进化（即 IL-bootstrapped RL 或 Demonstration-guided RL）

1. NVIDIA Isaac Lab (内置 RL 与 IL 混合训练管线)

作为目前灵巧手 Sim-to-Real 的行业标准，Isaac Lab 原生支持通过人类演示数据（IL）来引导大规模并行强化学习（RL）。
• 核心机制： 提供了 Learning from Demonstrations (LfD) 的算法接口。支持读入人类遥操作（如通过手套、VR）录制的轨迹数据，先使用行为克隆（BC）或扩散策略（Diffusion Policy）初始化网络权重，再调用并行 PPO 算法在仿真中进行亿级数据的 RL 探索。
• 开源代码库： GitHub - isaac-sim/IsaacLab


2. Robomimic + RoboSuite (斯坦福大学维护)

这是专门为了将模仿学习与强化学习结合而设计的高度模块化框架。
• 核心机制： 它完整实现了“先IL后RL”的闭环。你可以先用它训练一个 BC（行为克隆）模型，然后将其无缝接入它的在线 RL 模块（如结合 SAC、PPO 算法），让机器人基于模仿的底子继续去探索环境。
• 开源代码库： GitHub - ARISE-Initiative/robomimic
• 参考算法模块： 重点查看代码库中的 Algorithms 目录，里面包含 BC（纯IL）、BC-RNN、以及配合 RL 提升（Imitation-Bootstrapped RL）的实现。

3. Bi-DexHands (双手指灵巧操作基准)

由清华大学等团队开源，是目前针对多指灵巧手操作最全面的强化学习与模仿学习混合基准。
• 核心机制： 框架内针对大量复杂的灵巧操作任务（如旋转魔方、双手协作接球、插拔），提供了人类示范数据集。它不仅支持从头训练 RL，还包含了专门的示范引导强化学习（Demonstration-Guided RL）算法（如无缝衔接人类轨迹的 PPO-LfD 和 DDPGfD）。
• 开源代码库： GitHub - PKU-EPIC/Bi-DexHands

1. 第一步：统一数据格式（h5/json）
利用公司现有的遥操作手柄、或者单目相机手势识别，记录人手操作的轨迹（包含每个时间戳下自研手各关节的角度、物体的状态）。将这些数据导出为 Robomimic 兼容的 .hdf5 格式。
2. 第二步：使用 Robomimic 进行离线 IL 筑基
不用开复杂的物理仿真。直接使用 Robomimic 读入 .hdf5 数据，跑通一个 Behavior Cloning (BC) 策略。检查训练出的模型在特定起始位置下，是否能摆出“像人一样的抓握姿势”。
3. 第三步：导入 Isaac Lab 进行在线 RL 进化
将自研手的 URDF 导入 Isaac Lab，把第二步训练好的 BC 模型参数作为强化学习策略网络（如 PPO Actor）的初始权重（Pre-trained Checkpoint），开启物理仿真环境的在线 RL 训练，通过奖励函数（Reward Function）让它在碰撞、摩擦中自我进化。







