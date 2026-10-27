from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def tj_right_hand_cube_rotate_ppo_cfg() -> RslRlOnPolicyRunnerCfg:

    return RslRlOnPolicyRunnerCfg(

        actor=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=0.7,
        ),

        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256),
            activation="elu",
            obs_normalization=True,
            stochastic=False,
            init_noise_std=0.7,
        ),

        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # 10-19_18-52 run 的教训:0.008 恒定不衰减,3000 iteration 后
            # mean_noise_std 单调涨到 1.76、entropy 涨到 38,策略从未收敛
            # (rotation_progress 10800 峰值 0.202 后一路跌到 0.077)。熵奖励
            # 对"逃逸局部最优"有效,但必须回落才能收敛。0.008 已完成它的
            # 使命(口袋/卡死行为消失),现在从 model_10800 resume 并用
            # 0.003 让 std 降下来、策略真正定型。
            # entropy_coef=0.008,
            # 回退到蓝色 2026-10-21_15-58-28 基线:该轮是唯一验证过的稳定
            # 起点(净转角 ~3.05 rad/ep, cube_stalled=0)。黄色 10-22 轮把它
            # 抬到 0.001 并同时改了 6 个奖励参数,无法归因。
            # 10-26 轮:0.003 -> 0.0005。蓝色 std 停在 1.57 而落不下来,是因为
            # 从 model_19700 全量 resume 会把 log_std=1.57 一起带进来,0.003 的
            # 熵奖励只能让它停在原地小幅抖动。改为 scripts/train_blue_warmstart.py
            # 只拿权重、std 重置为 0.7,配合 0.0005 才有机会回到 1.0 以下,
            # KL 才会正到让 adaptive lr 离开 1e-5 地板。
            entropy_coef=0.0005, # 0.003(蓝色 resume) -> 0.0005(蓝轮超参)
            num_learning_epochs=5,
            # 4 -> 2:4090 上网络更新不是瓶颈(bs=16384 时 fwd+bwd 约 23ms/iter,
            # bs=32768 约 48ms),省下的时间用来同卡并行跑多个 seed。
            num_mini_batches=2, # 4(蓝色)
            # 同蓝色基线。注意 adaptive 分支每次都把 param_group lr 重写为
            # self.learning_rate,所以 resume 后实际从 3e-4 起步;但 std 仍
            # 偏大时会立刻被 KL 打到 1e-5 地板(蓝色 5000 iter 全程如此)。
            # desired_kl 保持 0.02:KL 公式衡量的是 std 增大,门槛调到 0.05 也不会
            # 让 lr 恢复,真正的前提是 std 降到 1.0 以下。
            learning_rate=3e-4, # 5e-4(黄色) -> 3e-4(蓝色)
            schedule="adaptive",
            gamma=0.995,#0.99
            lam=0.95,
            desired_kl=0.02, # 0.01
            max_grad_norm=1.0,
        ),

        experiment_name="tj_right_hand_cube_rotate",

        save_interval=100,

        num_steps_per_env=32,

        max_iterations=5_000,

        clip_actions=1.0,
    )


tj_right_hand_cube_rotate_rl_cfg = tj_right_hand_cube_rotate_ppo_cfg