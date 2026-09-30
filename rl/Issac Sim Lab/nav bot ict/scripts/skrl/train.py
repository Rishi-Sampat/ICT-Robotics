# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from unittest import runner

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import ict_bot_nav.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # runner = Runner(env, agent_cfg)
    # env.unwrapped.runner = runner

    import torch
    from skrl.agents.torch.ppo import PPO_RNN, PPO_CFG
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from cnn_gru import SharedModel

    device = env.device
    num_envs = env.num_envs

    mcfg = agent_cfg.get("model", {})
    seq_len = mcfg.get("sequence_length", 32)

    shared_model = SharedModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        num_envs=num_envs,
        num_layers=mcfg.get("num_layers", 1),
        hidden_size=mcfg.get("hidden_size", 256),
        sequence_length=seq_len,
    )

    models = {"policy": shared_model, "value": shared_model}

    acfg = agent_cfg["agent"]
    rollouts = acfg["rollouts"]

    memory = RandomMemory(memory_size=rollouts, num_envs=num_envs, device=device)

    cfg = PPO_CFG()
    cfg.rollouts = rollouts
    cfg.learning_epochs = acfg["learning_epochs"]
    cfg.mini_batches = acfg["mini_batches"]
    cfg.discount_factor = acfg["discount_factor"]
    cfg.lambda_ = acfg["lambda"]   # verify exact attribute name — may be `lambda_` since `lambda` is reserved
    cfg.learning_rate = acfg["learning_rate"]
    cfg.learning_rate_scheduler = KLAdaptiveLR
    cfg.learning_rate_scheduler_kwargs = acfg["learning_rate_scheduler_kwargs"]
    cfg.state_preprocessor = RunningStandardScaler
    cfg.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    cfg.value_preprocessor = RunningStandardScaler
    cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
    cfg.grad_norm_clip = acfg["grad_norm_clip"]
    cfg.ratio_clip = acfg["ratio_clip"]
    cfg.value_clip = acfg["value_clip"]
    cfg.clip_predicted_values = acfg["clip_predicted_values"]
    cfg.entropy_loss_scale = acfg["entropy_loss_scale"]
    cfg.value_loss_scale = acfg["value_loss_scale"]
    cfg.rewards_shaper = None
    cfg.time_limit_bootstrap = acfg["time_limit_bootstrap"]
    cfg.experiment.directory = log_root_path
    cfg.experiment.experiment_name = log_dir
    cfg.experiment.write_interval = "auto"
    cfg.experiment.checkpoint_interval = acfg["experiment"]["checkpoint_interval"]

    agent = PPO_RNN(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # --- shim so the logging block below (which expects `runner.agent`, `runner.run`) still works ---
    class _RunnerShim:
        def __init__(self, agent, env, timesteps):
            self.agent = agent
            self.env = env
            self.timesteps = timesteps
        def run(self):
            trainer = SequentialTrainer(
                cfg={"timesteps": self.timesteps, "headless": True},
                env=self.env,
                agents=self.agent,
            )
            trainer.train()

    runner = _RunnerShim(agent, env, agent_cfg["trainer"]["timesteps"])
    env.unwrapped.runner = runner


    # ---- TRAINING LOGS ----
    import torch

    ep_rewards    = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_lengths    = torch.zeros(env_cfg.scene.num_envs, device=env_cfg.sim.device)
    ep_count      = 0
    # Expanded to 9 to cover levels 0 through 8
    ep_level_counts = torch.zeros(9, device=env_cfg.sim.device) 

    original_record_transition = runner.agent.record_transition

    def custom_record_transition(**kwargs):
        nonlocal ep_rewards, ep_lengths, ep_count

        terminated = kwargs.get("terminated")
        truncated  = kwargs.get("truncated")
        rewards    = kwargs.get("rewards")
        timestep   = kwargs.get("timestep")
        timesteps  = kwargs.get("timesteps")

        dones_pre = (terminated | truncated).squeeze(-1)

        if dones_pre.any():
            timeout_snapshot    = truncated.squeeze(-1)[dones_pre].float().clone()
            terminated_snapshot = terminated.squeeze(-1)[dones_pre].float().clone()
        else:
            timeout_snapshot    = None
            terminated_snapshot = None

        original_record_transition(**kwargs)

        ep_rewards += rewards.squeeze(-1)
        ep_lengths += 1

        dones = (terminated | truncated).squeeze(-1)

        if dones.any():
            if not hasattr(custom_record_transition, "_buf"):
                custom_record_transition._buf = {
                    "rewards":     [],
                    "lengths":     [],
                    "timeouts":    [],
                    "stagnations": [],
                    "obs_level":   [],
                }

            buf = custom_record_transition._buf
            obs_level = getattr(env, "_obs_curr_level", 0)

            buf["rewards"].append(ep_rewards[dones].clone())
            buf["lengths"].append(ep_lengths[dones].clone())
            buf["timeouts"].append(timeout_snapshot)
            buf["stagnations"].append(terminated_snapshot)
            buf["obs_level"].append(
                torch.full(
                    (dones.sum().item(),), obs_level,
                    dtype=torch.float32, device=env_cfg.sim.device
                )
            )

            ep_count          += dones.sum().item()
            ep_rewards[dones]  = 0.0
            ep_lengths[dones]  = 0.0
            
            # Safety clamp to prevent out-of-bounds indexing
            safe_level = min(obs_level, 8) 
            ep_level_counts[safe_level] += dones.sum().item()

        if timestep % 500 == 0:
            buf = getattr(custom_record_transition, "_buf", {})

            level_labels = {
                0: "L0 — 0 obs at 0.0 m/s",
                1: "L1 — 1 obs at 0.0 m/s",
                2: "L2 — 2 obs at 0.1 m/s",
                3: "L3 — 3 obs at 0.3 m/s",
                4: "L4 — 4 obs at 0.5 m/s",
                5: "L5 — 5 obs at 0.8 m/s",
                6: "L6 — 6 obs at 1.0 m/s",
                7: "L7 — 7 obs at 1.3 m/s",
                8: "L8 — 8 obs at 1.5 m/s",
            }

            curr_level  = getattr(env, "_obs_curr_level", 0)
            curr_event  = getattr(env, "_obs_curr_last_event", "none")
            curr_sr_buf = getattr(env, "_obs_curr_successes", [])
            curr_sr_val = (
                sum(curr_sr_buf[-2000:]) / len(curr_sr_buf[-2000:])
                if len(curr_sr_buf) >= 200 else 0.0
            )
            curr_cooldown = getattr(env, "_obs_curr_cooldown", 0)
            curr_consec   = getattr(env, "_obs_curr_consecutive", 0)

            # Obstacle manager live state
            obs_mgr = getattr(env, "obstacle_manager", None)
            live_count  = getattr(obs_mgr, "_obstacle_count", 0) if obs_mgr else 0
            live_speed  = getattr(obs_mgr, "_max_speed", 0.0)    if obs_mgr else 0.0

            print("\n" + "=" * 70)
            print(f"  Timestep            : {timestep:>8} / {timesteps}  "
                f"({100 * timestep / timesteps:.1f}%)")
            print(f"  Episodes finished   : {int(ep_count)}")

            if buf.get("rewards"):
                all_r  = torch.cat(buf["rewards"])
                all_l  = torch.cat(buf["lengths"])
                all_to = torch.cat(buf["timeouts"])
                all_st = torch.cat(buf["stagnations"])

                step_dt      = env_cfg.sim.dt * env_cfg.decimation
                mean_ep_s    = all_l.mean().item() * step_dt
                timeout_rt   = all_to.mean().item()
                terminate_rt = all_st.mean().item()

                print("-" * 70)
                print(f"  Mean Reward         : {all_r.mean().item():>8.3f}   ↑ want rising")
                print(f"  Max  Reward         : {all_r.max().item():>8.3f}")
                print(f"  Min  Reward         : {all_r.min().item():>8.3f}")
                print("-" * 70)
                print(f"  Mean Ep Duration    : {mean_ep_s:>7.1f} s")
                print(f"  Timeout Rate        : {timeout_rt:>7.1%}   ↑ want high")
                print(f"  Termination Rate    : {terminate_rt:>7.1%}   ↓ want low (collision)")
                print("-" * 70)
                print("  CURRICULUM")
                print(f"    Level             : {curr_level}  {level_labels.get(curr_level, '')}")
                print(f"    Success Rate      : {curr_sr_val:>7.1%}  (rolling 1000 eps)")
                print(f"    Consecutive wins  : {curr_consec}")
                print(f"    Cooldown remaining: {curr_cooldown} eps")
                print(f"    Last event        : {curr_event}")
                print("-" * 70)
                print("  ACTIVE OBSTACLES")
                print(f"    Total             : {live_count}")
                print(f"    Max speed         : {live_speed:.2f} m/s")
                print("-" * 70)
                print("  Level distribution  :")
                
                # Update to 9 so it iterates 0 through 8 inclusive
                for lvl in range(9):
                    frac   = (
                        ep_level_counts[lvl] /
                        ep_level_counts.sum().clamp(min=1)
                    ).item()
                    bar    = "█" * int(frac * 20) if frac > 0.0 else ""
                    active = " ◄" if lvl == curr_level else ""
                    print(f"    {level_labels.get(lvl, f'L{lvl}'):24s}: "
                            f"{frac:>5.1%}  {bar}{active}")
                print("-" * 70)

                # Warnings
                if terminate_rt > 0.5:
                    print("  ⚠ termination > 50% — robot dying too often")
                if curr_sr_val < 0.1 and curr_level > 0:
                    print("  ⚠ success < 10% at non-zero level — consider demotion")
                if timeout_rt < 0.5:
                    print("  ⚠ timeout rate low — robot terminating frequently")

                for key in buf:
                    buf[key] = []

            else:
                print("  No episodes finished yet in this window")

            print("=" * 70 + "\n")

    runner.agent.record_transition = custom_record_transition
    # ---- END TRAINING LOGS ----


    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
