"""Warm-start TJ right-hand cube rotation from the blue 2026-10-21_15-58-28 run.

Why not ``scripts/train.py --agent.resume``: ``runner.load`` restores the
actor's learned ``std`` (1.57 in the blue run) and the Adam moments together
with the weights. The new ``entropy_coef`` would then decay that std from 1.57
again -- the same trajectory as the blue run, so nothing changes. The blue
policy stagnated because the action noise stayed above 1.0, which keeps the
adaptive-KL branch in its "learning rate too high" arm (lr pinned at 1e-5).

This script therefore keeps only the actor/critic *weights* and resets:
  - the action-noise std to the config's ``init_noise_std`` (0.7),
  - the optimizer (zero Adam moments),
  - the iteration counter (new ``model_XXXXX.pt`` numbering from 0).

The reward configuration is deliberately untouched: it is the blue baseline,
so that the only variables between blue and this run are entropy_coef,
num_mini_batches and the std reset.

Usage:
    uv run python scripts/train_blue_warmstart.py --seed 0
    uv run python scripts/train_blue_warmstart.py --seed 100 --num-envs 2048

Note on the wandb patch: rsl_rl's Logger hardcodes
``wandb.Settings(start_method="thread")``, which wandb >= 0.26 rejects. Apply
the fix from MyReadMe.md (``wandb.Settings()``) after any ``uv sync``.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


TASK_ID = "Mjlab-TJ-Right-HandCube-Rotate"
BLUE_CHECKPOINT = (
    "logs/rsl_rl/tj_right_hand_cube_rotate/2026-10-21_15-58-28/model_19700.pt"
)


@dataclass(frozen=True)
class WarmStartConfig:
  checkpoint: str = BLUE_CHECKPOINT
  seed: int = 0
  num_envs: int = 2048
  max_iterations: int = 3_000
  save_interval: int = 100
  gpu_id: int = 0
  run_tag: str = "blue_warm"


def _prepare_agent_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
  """Same conversion as scripts/train.py: dataclass -> runner dict."""
  from dataclasses import asdict

  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


def load_weights_only(
  runner: OnPolicyRunner, checkpoint: Path, init_noise_std: float
) -> None:
  """Copy actor/critic weights, then reset optimizer, std and iteration."""
  loaded = torch.load(str(checkpoint), weights_only=False, map_location="cpu")
  alg = runner.alg

  # The std parameter is deliberately NOT loaded: it stays at the freshly
  # initialized value so the new entropy_coef can decay it from there.
  std_param = "log_std" if hasattr(alg.actor, "log_std") else "std"
  actor_state = {
    k: v for k, v in loaded["actor_state_dict"].items() if k != std_param
  }
  missing, unexpected = alg.actor.load_state_dict(actor_state, strict=False)
  if unexpected or missing != [std_param]:
    raise RuntimeError(
      f"Unexpected actor-state mismatch: missing={missing}, unexpected={unexpected}"
    )
  alg.critic.load_state_dict(loaded["critic_state_dict"])

  # Zero Adam moments: the old ones point at the old (blue) value landscape.
  alg.optimizer = type(alg.optimizer)(
    chain(alg.actor.parameters(), alg.critic.parameters()),
    lr=alg.learning_rate,
  )

  value = math.log(init_noise_std) if std_param == "log_std" else init_noise_std
  with torch.no_grad():
    getattr(alg.actor, std_param).fill_(value)  # type: ignore[union-attr]

  runner.current_learning_iteration = 0
  print(
    f"[INFO] weights-only warm start from {checkpoint}: "
    f"actor/critic weights kept; std reset to {init_noise_std:g}; "
    "optimizer and iteration counter reset"
  )


def run(cfg: WarmStartConfig) -> None:
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu_id))
  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  checkpoint = Path(cfg.checkpoint).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  env_cfg = load_env_cfg(TASK_ID)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  agent_cfg = load_rl_cfg(TASK_ID)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  # Fresh seed per run so a "no growth" result cannot be blamed on the
  # blue run's seed=1.
  agent_cfg.seed = cfg.seed
  agent_cfg.max_iterations = cfg.max_iterations
  agent_cfg.save_interval = cfg.save_interval
  agent_cfg.run_name = f"{cfg.run_tag}_s{cfg.seed}"

  log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
  run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{agent_cfg.run_name}"
  log_dir = log_root / run_name
  print(f"[INFO] log dir: {log_dir}")

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  agent_dict = _prepare_agent_cfg(agent_cfg)
  runner = OnPolicyRunner(env, agent_dict, str(log_dir), "cuda:0")
  load_weights_only(runner, checkpoint, agent_cfg.actor.init_noise_std)

  (log_dir / "params").mkdir(parents=True, exist_ok=True)
  from dataclasses import asdict

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)

  runner.learn(
    num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True
  )
  env.close()


def main() -> None:
  import in_hand_rotation_mjlab.tasks  # noqa: F401  (populate task registry)

  cfg = tyro.cli(WarmStartConfig, config=(tyro.conf.FlagConversionOff,))
  run(cfg)


if __name__ == "__main__":
  main()
