import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import wandb
import os

import torch
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import gymnasium as gym

from tensordict import TensorDict


from policies import BasePolicy
from utils.replay_memory import ReplayMemory

import matplotlib.pyplot as plt
from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logger.info('Initialising Environment')

    env : gym.Env = hydra.utils.instantiate(cfg.env)

    logger.info('Initialising Policy')
    cfg.num_steps_per_update = int((cfg.policy.frames_per_batch / cfg.policy.alpha_batch) / cfg.policy.alpha_sub_batch)
    cfg.num_updates = cfg.num_steps // cfg.num_steps_per_update
    cfg.num_episodes = None

    policy : BasePolicy = hydra.utils.instantiate(cfg.policy, env=env, device=device)
    if cfg.scheduler != {}:
        cfg.scheduler.T_max = cfg.num_updates
        scheduler : LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=policy.optimizer)
    else:
        scheduler = None

    run_path = None
    if cfg.wandb.log:
        logger.info('Initialising Weights and Biases')
        wandb_config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(config=wandb_config, project=cfg.wandb.project, settings=wandb.Settings(start_method="thread"))
        wandb.define_metric("num_updates")
        wandb.define_metric("num_env_eps")
        wandb.define_metric("num_env_steps")
        wandb.define_metric("CVaR/*", step_metric="num_env_steps")
        wandb.define_metric("Policy/*", step_metric="num_env_steps")

        run_path = f'saved_policies/{wandb.run.name}'
        if cfg.save_model and not os.path.exists(run_path):
                logger.info(f'Saving to: {run_path}')
                os.makedirs(run_path)


    logger.info('Training')
    train(policy=policy, env=env, cfg=cfg, scheduler=scheduler, run_path=run_path)

    logger.info('Finished Run')
    if cfg.wandb.log:
        wandb.finish()



def train(policy: BasePolicy, env: gym.Env, cfg: DictConfig, scheduler : LRScheduler, run_path: str = None):
    total_episodes = 0
    return_cap = cfg.initial_return_cap
    last_batch_returns = []

    replay_memory = ReplayMemory(
        frames_per_batch=cfg.policy.frames_per_batch, 
        alpha_batch=cfg.policy.alpha_batch,
        alpha_sub_batch=cfg.policy.alpha_sub_batch,
        device=device,
    )

    for i_update in range(cfg.num_updates):
        # Gather Batch Trajectories
        episode_memory_full = False
        while not episode_memory_full:
            total_reward_uncapped = 0
            if cfg.cap_return:
                total_reward = min(total_reward_uncapped, return_cap)
            else:
                total_reward = total_reward_uncapped
            state, info = env.reset()
            state, mask = torch.tensor(state, device=device), torch.tensor(info["mask"], dtype=torch.bool, device=device)
            done = False

            while not done:
                action, log_prob = policy.select_action(state.unsqueeze(0), mask.unsqueeze(0))

                next_state, reward, done, info = env.step(action=action.item())
                next_state, next_mask = torch.tensor(next_state, device=device), torch.tensor(info["mask"], dtype=torch.bool, device=device)

                total_reward_uncapped += reward
                if cfg.cap_return:
                    reward = min(total_reward_uncapped, return_cap) - total_reward
                total_reward += reward

                replay_memory.append_frame(
                    TensorDict({
                        'observation' : state.unsqueeze(0), 
                        'action' : action, 
                        'sample_log_prob' : log_prob, 
                        'next' : {
                            'observation' : next_state.unsqueeze(0), 
                            'reward' : torch.tensor([reward], dtype=torch.float32, device=device),
                            'mask' : next_mask.unsqueeze(0),
                            'done': torch.tensor([done], dtype=torch.bool, device=device)
                        },
                        'mask' : mask.unsqueeze(0),
                    }, [1])
                )

                state = next_state
                mask = next_mask
            
            last_batch_returns.append(total_reward_uncapped)
            episode_memory_full = replay_memory.finish_episode(final_return=total_reward)
            total_episodes += 1

            logger.debug(f"Run Return: {total_reward*cfg.env.reward_normalisation}")

        # Train on Gathered Trajectories
        logger.info(f'{(i_update+1) * cfg.num_steps_per_update}/{cfg.num_steps}')

        tensordict_data = replay_memory.get_tensordict()
        policy.train(tensordict_data)        

        sorted_last_batch_returns = sorted(last_batch_returns)

        wandb_log_dict = {"num_env_steps" : (i_update + 1) * cfg.num_steps_per_update,
                            "num_env_eps" : total_episodes,
                            "num_updates" : i_update + 1}

        for alpha in cfg.cvar_test_values:
            last_batch_cvar = np.mean(sorted_last_batch_returns[:max(int(len(last_batch_returns) * alpha), 1)])
            logger.debug(sorted_last_batch_returns)
            wandb_log_dict[f'CVaR/CVaR: {alpha}'] = last_batch_cvar*cfg.env.reward_normalisation

        if cfg.cap_return:
            new_cap = sorted_last_batch_returns[int(len(last_batch_returns) * cfg.cap_alpha)]
            return_cap += cfg.cap_tau * (new_cap - return_cap)
            return_cap = np.maximum(return_cap, cfg.minimum_return_cap)
            wandb_log_dict['Policy/Return Cap'] = return_cap*cfg.env.reward_normalisation

        if scheduler != None:
            wandb_log_dict['Policy/Learning Rate'] = scheduler.get_last_lr()[0]
            scheduler.step()

        logger.debug(wandb_log_dict)
        if cfg.wandb.log:
            wandb.log(data=wandb_log_dict)

            if cfg.save_model:
                policy.save(run_path)

        last_batch_returns = []


if __name__ == "__main__":
    main()