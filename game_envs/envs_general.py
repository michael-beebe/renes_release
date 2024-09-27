# from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from copy import deepcopy

import gym
import numpy as np
import tensorly as tl
import torch
from gym import spaces

# from gym.spaces.box import Box
# from gym.wrappers.clip_action import ClipAction
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
#     WarpFrame,
# )
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnvWrapper

from game2graph.response_graphs import (
    preprocess_response_graph_to_gnn_no_max,
    gen_response_graph,
)
from game_envs.subproc import SubprocVecEnv
from solvers import (
    alpharank_strategy,
    projected_replicator_dynamics,
    fictitious_play_strategy,
    ce_strategy,
    # cce_strategy,
)
from solvers.eval import nash_conv
from utils import normalize_tables


# from game2graph.game_data import gen_response_graph


class game_env(gym.Env):
    def __init__(self, game_dataset, args, is_train=True):
        self.args = args
        self.game_dataset = game_dataset
        self.meta_solver = args.meta_solver

        self.original_game = None
        self.current_game = None
        self.original_gnn_data = None
        self.current_gnn_data = None

        self.weights = None
        self.factors = None
        self.normalizer = -10000
        self.pre_nc = -1000
        self.current_step = 0
        self.min_nc = 1000
        self.init_nc = 1000

        self.is_train = is_train
        self.current_game_idx = 0
        self.eval_val = 0
        self.eval_abs_val = 0

        payoff_tables, weights, factors = self.game_dataset[0]
        self.observation_space = spaces.Box(
            low=args.min_val,
            high=args.max_val,
            shape=(2 * len(payoff_tables.flatten()),),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(args.action_size,), dtype=np.float32
        )

    def payoff_table_to_gnn(self, payoff_tables):
        response_graph_dict = gen_response_graph(payoff_tables, self.args)
        response_graph_dict["payoff_tables"] = payoff_tables
        gnn_data = preprocess_response_graph_to_gnn_no_max(response_graph_dict)
        return gnn_data

    def reset(self):
        self.current_step = 0
        if self.is_train:
            idx = np.random.choice(range(len(self.game_dataset)))
        else:
            idx = self.current_game_idx
            self.current_game_idx += 1
            self.current_game_idx %= len(self.game_dataset)
        payoff_tables, weights, factors = self.game_dataset[idx]
        self.original_game = deepcopy(payoff_tables)
        self.current_game = deepcopy(payoff_tables)
        # print(self.original_game)
        self.weights = weights
        self.factors = factors
        init_nc = self._eval()
        self.pre_nc = init_nc
        self.min_nc = init_nc
        self.init_nc = init_nc
        if self.is_train:
            # if init_nc < 0.5:
            #     self.normalizer = 1.0
            # else:
            #     self.normalizer = init_nc
            self.normalizer = 1.0
        else:
            self.normalizer = init_nc + 1e-6
        # obs = np.concatenate([deepcopy(self.original_game).flatten(), deepcopy(self.current_game).flatten()])
        # print(obs.shape)
        # print(deepcopy(self.original_game).flatten(),)
        self.original_gnn_data = self.payoff_table_to_gnn(self.original_game)
        self.current_gnn_data = self.original_gnn_data
        return {
            "games": [deepcopy(self.original_game), deepcopy(self.current_game)],
            "gnn": [
                deepcopy(self.original_gnn_data),
                deepcopy(self.current_gnn_data),
            ],
        }

    def _eval(self):
        if self.meta_solver == "alpha_rank":
            res = alpharank_strategy(self.current_game, return_joint=False)
        elif self.meta_solver == "fp":
            res = fictitious_play_strategy(self.current_game, max_iterations=int(5e3))
        elif self.meta_solver == "ce":
            res = ce_strategy(self.current_game, iterations=int(5e3))
        # elif self.meta_solver == "cce":
        #     res = cce_strategy(self.current_game)
        else:
            res = projected_replicator_dynamics(
                payoff_tensors=self.current_game, prd_iterations=int(5e3)
            )
        return sum(nash_conv(self.original_game, res))

    def step(self, action):
        args = self.args
        factors = self.factors
        self.current_step += 1
        mod_weights = args.weight_step * action
        self.current_game += tl.cp_to_tensor((mod_weights, factors))
        self.current_game = normalize_tables(
            tables=self.current_game, min_val=args.min_val, max_val=args.max_val
        )

        current_nc = self._eval()
        if self.current_step >= args.max_steps:
            done = True
        else:
            done = False
        if current_nc < self.min_nc:
            self.min_nc = current_nc
        reward = (self.pre_nc - current_nc) / self.normalizer
        # print("current nc:{}".format(current_nc))
        # print(self.pre_nc)
        # print(current_nc)
        # print("reward: {}".format(reward))
        self.pre_nc = current_nc

        info = {}
        if done:
            episode_min = (self.init_nc - self.min_nc) / self.normalizer
            episode_abs_min = self.init_nc - self.min_nc
            info["episode"] = {
                "r": episode_min,
                "abs_r": episode_abs_min,
                "init_nc": self.init_nc,
                "min_nc": self.min_nc,
            }
            if not self.is_train:
                self.eval_val += episode_min
                self.eval_abs_val += episode_abs_min

            if (not self.is_train) and (self.current_game_idx == 0):
                info["eval"] = self.eval_val / (len(self.game_dataset))
                info["abs_eval"] = self.eval_abs_val / (len(self.game_dataset))
                self.eval_val = 0
                self.eval_abs_val = 0

        self.current_gnn_data = self.payoff_table_to_gnn(self.current_game)
        return (
            {
                "games": [deepcopy(self.original_game), deepcopy(self.current_game)],
                "gnn": [
                    deepcopy(self.original_gnn_data),
                    deepcopy(self.current_gnn_data),
                ],
            },
            reward,
            done,
            info,
        )

    def render(self, mode="human"):
        pass


def make_env(game_dataset, args, is_train=True):
    def _thunk():
        env = game_env(game_dataset=game_dataset, args=args, is_train=is_train)

        return env

    return _thunk


def make_vec_envs(game_dataset, num_processes, args, device, is_train=True):
    size_for_each_env = len(game_dataset) // num_processes
    assert num_processes > 1
    envs = [
        make_env(
            game_dataset=game_dataset[
                size_for_each_env * i : size_for_each_env * (i + 1)
            ],
            args=args,
            is_train=is_train,
        )
        for i in range(num_processes)
    ]

    envs = SubprocVecEnv(envs)
    # if len(envs) > 1:
    #     envs = SubprocVecEnv(envs)
    # else:
    #     envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        # print("vec pytorch: {}".format(obs.shape))
        # obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
