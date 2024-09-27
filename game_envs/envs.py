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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

from solvers import (
    alpharank_strategy,
    projected_replicator_dynamics,
    fictitious_play_strategy,
    ce_strategy,
    # cce_strategy,
)
from solvers.eval import nash_conv
from utils import normalize_tables


class game_env(gym.Env):
    def __init__(self, game_dataset, args, is_train=True):
        self.args = args
        self.game_dataset = game_dataset
        self.meta_solver = args.meta_solver

        self.original_game = None
        self.current_game = None
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
            self.normalizer = init_nc + 1e-8
        # obs = np.concatenate([deepcopy(self.original_game).flatten(), deepcopy(self.current_game).flatten()])
        # print(obs.shape)
        return np.concatenate(
            [
                deepcopy(self.original_game).flatten(),
                deepcopy(self.current_game).flatten(),
            ]
        )

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
                payoff_tensors=self.current_game, prd_iterations=int(2e3)
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
            # info["episode"]["r"] = episode_min
            # info["episode"]["abs_r"] = episode_abs_min
            if not self.is_train:
                self.eval_val += episode_min
                self.eval_abs_val += episode_abs_min

            if (not self.is_train) and (self.current_game_idx == 0):
                info["eval"] = self.eval_val / (len(self.game_dataset))
                info["abs_eval"] = self.eval_abs_val / (len(self.game_dataset))
                self.eval_val = 0
                self.eval_abs_val = 0

        return (
            np.concatenate(
                [
                    deepcopy(self.original_game).flatten(),
                    deepcopy(self.current_game).flatten(),
                ]
            ),
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

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

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
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# from game2graph.game_data import gen_game_datasets
# from configs import get_parser
#
# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     args.min_players = 2
#     args.max_players = 2
#     args.max_actions = 4
#     args.min_actions = 4
#     args.train_number = 30
#     args.test_number = 0
#     train_dataset, test_dataset = gen_game_datasets(args)
#
#     envs = make_vec_envs(game_dataset=train_dataset, args=args, num_processes=5)
#
#     print(envs.reset().shape)

# # Checks whether done was caused my timit limits or not
# class TimeLimitMask(gym.Wrapper):
#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         if done and self.env._max_episode_steps == self.env._elapsed_steps:
#             info["bad_transition"] = True
#
#         return obs, rew, done, info
#
#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)
#
#
# # Can be used to test recurrent policies for Reacher-v2
# class MaskGoal(gym.ObservationWrapper):
#     def observation(self, observation):
#         if self.env._elapsed_steps > 0:
#             observation[-2:] = 0
#         return observation
#
#
# class TransposeObs(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         """
#         Transpose observation space (base class)
#         """
#         super(TransposeObs, self).__init__(env)
#
#
# class TransposeImage(TransposeObs):
#     def __init__(self, env=None, op=[2, 0, 1]):
#         """
#         Transpose observation space for images
#         """
#         super(TransposeImage, self).__init__(env)
#         assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
#         self.op = op
#         obs_shape = self.observation_space.shape
#         self.observation_space = Box(
#             self.observation_space.low[0, 0, 0],
#             self.observation_space.high[0, 0, 0],
#             [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
#             dtype=self.observation_space.dtype,
#         )
#
#     def observation(self, ob):
#         return ob.transpose(self.op[0], self.op[1], self.op[2])
#
#
# class VecPyTorch(VecEnvWrapper):
#     def __init__(self, venv, device):
#         """Return only every `skip`-th frame"""
#         super(VecPyTorch, self).__init__(venv)
#         self.device = device
#         # TODO: Fix data types
#
#     def reset(self):
#         obs = self.venv.reset()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         return obs
#
#     def step_async(self, actions):
#         if isinstance(actions, torch.LongTensor):
#             # Squeeze the dimension for discrete actions
#             actions = actions.squeeze(1)
#         actions = actions.cpu().numpy()
#         self.venv.step_async(actions)
#
#     def step_wait(self):
#         obs, reward, done, info = self.venv.step_wait()
#         obs = torch.from_numpy(obs).float().to(self.device)
#         reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
#         return obs, reward, done, info
#
#
# class VecNormalize(VecNormalize_):
#     def __init__(self, *args, **kwargs):
#         super(VecNormalize, self).__init__(*args, **kwargs)
#         self.training = True
#
#     def _obfilt(self, obs, update=True):
#         if self.obs_rms:
#             if self.training and update:
#                 self.obs_rms.update(obs)
#             obs = np.clip(
#                 (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
#                 -self.clip_obs,
#                 self.clip_obs,
#             )
#             return obs
#         else:
#             return obs
#
#     def train(self):
#         self.training = True
#
#     def eval(self):
#         self.training = False
#
#
# # Derived from
# # https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
# class VecPyTorchFrameStack(VecEnvWrapper):
#     def __init__(self, venv, nstack, device=None):
#         self.venv = venv
#         self.nstack = nstack
#
#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]
#
#         low = np.repeat(wos.low, self.nstack, axis=0)
#         high = np.repeat(wos.high, self.nstack, axis=0)
#
#         if device is None:
#             device = torch.device("cpu")
#         self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)
#
#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype
#         )
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)
#
#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, : -self.shape_dim0] = self.stacked_obs[
#             :, self.shape_dim0 :
#         ].clone()
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0 :] = obs
#         return self.stacked_obs, rews, news, infos
#
#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0 :] = obs
#         return self.stacked_obs
#
#     def close(self):
#         self.venv.close()
