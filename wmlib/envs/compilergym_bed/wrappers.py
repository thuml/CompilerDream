# Note: this code wa copied from
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Environment wrappers to closer replicate the MLSys'20 Autophase paper."""
import traceback
import torch
from typing import Optional
from compiler_gym.wrappers import (
    ConstrainedCommandline,
    ObservationWrapper,
    RewardWrapper,
    CompilerEnvWrapper,
    ActionWrapper,
)
import copy
import os, json, pickle
from compiler_gym.util.gym_type_hints import ActionType, ObservationType
from compiler_gym.envs import CompilerEnv, LlvmEnv
from typing import *
from abc import ABC, abstractmethod
from compiler_gym.views import ObservationSpaceSpec, ObservationView, RewardView
from compiler_gym.spaces.reward import Reward
from compiler_gym.spaces import Discrete, Dict
from .llvm_programl_env import GraphSizeExceededException
import compiler_gym

import random
import gym
import numpy as np
import logging
from func_timeout import *
from . import utils

# this is a logger for printing exception in step() & close()
logger = logging.getLogger(__name__)


class StepCountWrapper(CompilerEnvWrapper, ABC):
    '''
    Add Step count to observation
    step count will be add to the end of observation vector as a float value
    it increases 1/max_step each step
    ranges in [0, 1]
    '''
    def __init__(self, env: CompilerEnv, max_step=45):
        super().__init__(env)
        self.step_count = 0.0
        self.max_step = max_step
        assert isinstance(self.observation_space_spec.space, gym.spaces.Box), \
            "Can only add step count to vector features"
        self.observation_len = self.env.observation_space_spec.space.shape[0]
        self.env.observation_space_spec.space = gym.spaces.Box(
            low=np.full(
                self.observation_len+1, 0, dtype=np.float32
            ),
            high=np.full(
                self.observation_len+1, 1, dtype=np.float32
            ),
            dtype=np.float32,
        )
    
    def fork(self):
        fork_env = StepCountWrapper(self.env.fork(), self.max_step)
        fork_env.step_count = self.step_count
        return fork_env

    def reset(self, *args, **kwargs):
        self.step_count = 0.0
        observation = self.env.reset(*args, **kwargs)
        return self.convert_observation(observation)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str,
                                                    ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str,
                                              ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        self.step_count += 1/self.max_step
        self.step_count = min(1.0, self.step_count)

        return self.convert_observation(observation), reward, done, info
    
    @property
    def stepCount(self):
        return round(self.step_count*self.max_step)

    def convert_observation(self, observation: ObservationType) -> ObservationType:
        """Translate an observation to the new space."""
        observation = np.concatenate(
            (observation, np.array([self.step_count], dtype=np.float32)))
        return observation


class O0BasedRewardWrapper(CompilerEnvWrapper):
    """
    Reward=(C_{t-1}-C_t)/C_0
    """
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        self.last_inst_cnt = 0
    
    def fork(self):
        fork_env = type(self)(env = self.env.fork())
        fork_env.last_inst_cnt = self.last_inst_cnt
        return fork_env        

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        return ret

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        if reward is not None:
            if self.episode_reward is not None:
                self.unwrapped.episode_reward -= reward
                reward = self.convert_reward(reward)
                self.unwrapped.episode_reward += reward
            else:
                reward = self.convert_reward(reward)
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        return observation, reward, done, info

    def convert_reward(self, reward):
        if self.env.observation["IrInstructionCountO0"] == 0:
            return 0.0
        return (self.last_inst_cnt - self.env.observation["IrInstructionCount"]) \
            / self.env.observation["IrInstructionCountO0"]

        # if self.env.observation["IrInstructionCountOz"] == 0:
        #     return 0.0
        # return (self.last_inst_cnt - self.env.observation["IrInstructionCount"]) \
        #     / self.env.observation["IrInstructionCountOz"]

class ReductionRewardWrapper(CompilerEnvWrapper):
    """
    Reward=C_z/C_t - C_z/C_{t-1}
    """
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        self.last_inst_cnt = 0
    
    def fork(self):
        fork_env = type(self)(env = self.env.fork())
        fork_env.last_inst_cnt = self.last_inst_cnt
        return fork_env        

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        return ret

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        if reward is not None:
            if self.episode_reward is not None:
                self.unwrapped.episode_reward -= reward
                reward = self.convert_reward(reward)
                self.unwrapped.episode_reward += reward
            else:
                reward = self.convert_reward(reward)
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        return observation, reward, done, info

    def convert_reward(self, reward):
        return self.env.observation["IrInstructionCountOz"] / self.env.observation["IrInstructionCount"] \
            - self.env.observation["IrInstructionCountOz"] / self.last_inst_cnt

class BinaryRewardWrapper(CompilerEnvWrapper):
    """
    Reward in [-1, 0, +1]
    """
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        self.last_inst_cnt = 0
    
    def fork(self):
        fork_env = type(self)(env = self.env.fork())
        fork_env.last_inst_cnt = self.last_inst_cnt
        return fork_env        

    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        return ret

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        if reward is not None:
            if self.episode_reward is not None:
                self.unwrapped.episode_reward -= reward
                reward += utils.triplet_compare(self.env.observation["IrInstructionCountOz"], self.env.observation["IrInstructionCount"]) if done else 0
                self.unwrapped.episode_reward += reward
            else:
                reward += utils.triplet_compare(self.env.observation["IrInstructionCountOz"], self.env.observation["IrInstructionCount"]) if done else 0
        self.last_inst_cnt = self.env.observation["IrInstructionCount"]
        # utils.ForkedPdb().set_trace()
        return observation, reward, done, info
    
class OzOptActionWrapper(CompilerEnvWrapper):
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        self.random_code = random.randint(0, 1000000000)
        self.temp_file_list = []
        n_action = self.action_space.n+1
        self.origin_action_space_oz = copy.copy(self.action_space)
        self.action_space = Discrete(n_action, "action-space-with-Oz-opt-action")
        self.baseline = None
        if not os.path.exists("temp"):
            os.mkdir("temp")

    def reset(self, *args, **kwargs) -> ObservationType:
        obs = super().reset(*args, **kwargs)
        self.baseline = (self.env.observation['IrInstructionCountOz'], self.env.observation['IrInstructionCountO0'])
        for file in self.temp_file_list:
            os.remove(file)
        self.temp_file_list = []
        self.original_benchmark = copy.deepcopy(super().benchmark)
        return obs
    
    def fork(self):
        fork_env = type(self)(env = self.env.fork())
        fork_env.baseline = self.baseline
        
    @property
    def observation(self) -> ObservationView:
        class TempDict:
            def __init__(self, env, baseline) -> None:
                self.dict = env.observation
                self.baseline = baseline
                
            def __getitem__(self, item):
                if item == "IrInstructionCountOz":
                    return self.baseline[0]
                if item == "IrInstructionCountO0":
                    return self.baseline[1]
                return self.dict[item]
        return TempDict(self.env, self.baseline)
    
    @property
    def benchmark(self):
        return self.original_benchmark
        
    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        if self.action_space.n-1 not in actions:
            return self.env.multistep(actions, observation_spaces, reward_spaces, observations, rewards)
        tot_rwd = 0
        for action in actions:
            if action != self.action_space.n-1:
                tot_rwd += self.env.multistep([action], observation_spaces, reward_spaces, observations, rewards)[1]
            else:
                last_inst_cnt = self.env.observation["IrInstructionCount"]
                
                bc_path = self.env.observation["BitcodeFile"]
                temp_bc_file = f"{os.path.join('./temp', os.path.basename(bc_path).split('.')[0])}-opt-{self.random_code:09d}.bc"
                os.system(f"./bin/opt -Oz {bc_path} -o {temp_bc_file}")
                # print(f"opt -Oz {bc_path} -o {temp_bc_file}")
                self.temp_file_list.append(temp_bc_file)
                observation = self.env.reset(benchmark=f"file://{temp_bc_file}")
                reward = (last_inst_cnt -self.env.observation["IrInstructionCount"]) / \
                    max(1, self.observation["IrInstructionCountO0"] - self.observation["IrInstructionCountOz"])
                return observation, tot_rwd + reward, True, {}
                # return observation, tot_rwd + reward, False, {}

    def close(self):
        for file in self.temp_file_list:
            os.remove(file)
        self.temp_file_list = []
        return super().close()

class RewardWrapperModified(CompilerEnvWrapper, ABC):
    """Wraps a :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
    to allow an reward space transformation.
    """

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        observation, reward, done, info = self.env.multistep(
            actions,
            observation_spaces=observation_spaces,
            reward_spaces=reward_spaces,
            observations=observations,
            rewards=rewards,
        )
        if reward is not None:
            if self.episode_reward is not None:
                self.unwrapped.episode_reward -= reward
                reward = self.convert_reward(reward)
                self.unwrapped.episode_reward += reward
            else:
                reward = self.convert_reward(reward)
        
        return observation, reward, done, info

    @abstractmethod
    def convert_reward(self, reward):
        """Translate a reward to the new space."""
        raise NotImplementedError


class ClampedReward(RewardWrapperModified):
    """A wrapper class that clamps reward signal within a bounded range,
    optionally with some leaking for out-of-range values.
    """

    def __init__(
        self,
        env: CompilerEnv,
        min: float = -1,
        max: float = 1,
        leakiness_factor: float = 0.001,
    ):
        super().__init__(env)
        self.min = min
        self.max = max
        self.leakiness_factor = leakiness_factor
    
    def fork(self):
        return type(self)(env = self.env.fork(), min = self.min,
                          max = self.max, 
                          leakiness_factor = self.leakiness_factor)

    def convert_reward(self, reward: float) -> float:
        if reward > self.max:
            ret = self.max + (reward - self.max) * self.leakiness_factor
        elif reward < self.min:
            ret = self.min + (reward - self.min) * self.leakiness_factor
        else:
            ret = reward
        return ret

class RuntimeTimeoutException(Exception):
    pass

class RuntimeReward(RewardWrapperModified):

    timeout = 10
    
    def __init__(
        self,
        env: CompilerEnv,
    ):
        super().__init__(env)
        self.init_runtime = 0.0
        self.runtime_repeat = 20
        
    def fork(self):
        fork_env =  type(self)(env = self.env.fork())
        fork_env.init_runtime = self.init_runtime
        return fork_env
    
    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.set_runtime_repeat()
        self.init_runtime = self.get_runtime()
        return ret

    def convert_reward(self, reward: float) -> float:
        self.current_runtime = self.get_runtime()
        ret = (self.init_runtime-self.current_runtime)/self.init_runtime
        return ret
    
    def get_runtime(self):
        try:
            return np.average(self.get_runtime_observation())
        except FunctionTimedOut:
            raise RuntimeTimeoutException("Runtime Timeout.")
    
    @func_set_timeout(timeout)
    def get_runtime_observation(self):
        return self.env.observation["Runtime"]
    
    def set_runtime_repeat(self) ->None:
        self.env.send_param(
                    "llvm.set_runtimes_per_observation_count", str(self.runtime_repeat)
                )


class AutophaseNormalizedFeatures(ObservationWrapper):
    """A wrapper for LLVM environments that use the Autophase observation space
    to normalize and clip features to the range [0, 1].
    """

    # The index of the "TotalInsts" feature of autophase.
    TotalInsts_index = 51

    def __init__(self, env: CompilerEnv):
        super().__init__(env=env)
        # Force Autophase observation space.
        self.env.observation_space = self.env.unwrapped.observation.spaces["Autophase"]
        # Adjust the bounds to reflect the normalized values.
        if isinstance(self.env.observation_space, gym.spaces.Box):
            autophase_space = self.env.observation_space
        elif isinstance(self.env.observation_space, gym.spaces.Dict): # TODO： this is ugly. figure out why unwrapped space are modified. (dcy)
            autophase_space = self.env.observation_space['autophase']
        new_space = gym.spaces.Box(
            low=np.full(
                autophase_space.shape[0], 0, dtype=np.float32
            ),
            high=np.full(
                autophase_space.shape[0], 1, dtype=np.float32
            ),
            dtype=np.float32,
        )
        self.env.observation_space_spec.space = new_space

    def convert_observation(self, observation):
        if observation[self.TotalInsts_index] <= 0:
            return np.zeros(observation.shape, dtype=np.float32)
        return np.clip(
            observation.astype(np.float32) /
            observation[self.TotalInsts_index], 0, 1
        )

class ObservationToDictionaryWrapper(ObservationWrapper):
    def __init__(self, env: CompilerEnv, oringin_obs_name="oringin"):
        super().__init__(env)
        self.obs_name = oringin_obs_name
        self.observation_space_spec.space = Dict(
            {oringin_obs_name:self.env.observation_space},
            "dict_obs"
        )
    
    def convert_observation(self, observation: ObservationType) -> ObservationType:
        obs = {self.obs_name: observation}
        return obs
    
    def fork(self):
        return type(self)(self.env.fork(), self.obs_name)

class ConcatActionsHistogram(ObservationWrapper):
    """A wrapper that concatenates a histogram of previous actions to each
    observation.

    The actions histogram is concatenated to the end of the existing 1-D box
    observation, expanding the space.

    The actions histogram has bounds [0,inf]. If you specify a fixed episode
    length `norm_to_episode_len`, each histogram update will be scaled by
    1/norm_to_episode_len, so that `sum(observation) == 1` after episode_len
    steps.
    """

    def __init__(self, env: CompilerEnv, norm_to_episode_len: int = 0):
        super().__init__(env=env)
        assert isinstance(
            self.observation_space, gym.spaces.Box
        ), f"Can only contatenate actions histogram to box shape, not {self.observation_space}"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Can only construct histograms from discrete spaces"
        assert len(
            self.observation_space.shape) == 1, "Requires 1-D observation space"
        self.norm_to_episode_len = norm_to_episode_len
        self.increment = 1 / norm_to_episode_len if norm_to_episode_len else 1
        # Reshape the observation space.
        self.env.observation_space_spec.space = gym.spaces.Box(
            low=np.concatenate(
                (
                    self.env.observation_space.low,
                    np.zeros(
                        self.action_space.n, dtype=self.env.observation_space.dtype
                    ),
                )
            ),
            high=np.concatenate(
                (
                    self.env.observation_space.high,
                    # The upper bound is 1.0 if we are normalizing to the
                    # episode length, else infinite for unbounded episode
                    # lengths.
                    np.ones(self.action_space.n,
                            dtype=self.env.observation_space.dtype)
                    * (1.0 if norm_to_episode_len else np.inf),
                )
            ),
            dtype=self.env.observation_space.dtype,
        )
        self.histogram = np.zeros(
            (self.action_space.n,), dtype=self.env.observation_space.dtype
        )
        

    def reset(self, *args, **kwargs):
        self.histogram = np.zeros(
            (self.action_space.n,), dtype=self.env.observation_space.dtype
        )
        return super().reset(*args, **kwargs)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        observations=None,
        **kwargs,
    ):
        for a in actions:
            self.histogram[a] += self.increment
        return super().multistep(actions, **kwargs)

    def convert_observation(self, observation):
        return np.concatenate((observation, self.histogram)).astype(
            self.env.observation_space.dtype
        )
    
    def fork(self):
        fork_env =  type(self)(env = self.env.fork(), 
                          norm_to_episode_len = self.norm_to_episode_len)
        fork_env.histogram = copy.deepcopy(self.histogram)
        return fork_env

class AppendActionsHistogram(ObservationWrapper):
    """
    Append actionhistogram to observation which is list type.
    Support both list and dict observation space.
    """

    def __init__(self, env: CompilerEnv, norm_to_episode_len: int = 0):
        super().__init__(env=env)
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Can only construct histograms from discrete spaces"
        self.norm_to_episode_len = norm_to_episode_len
        self.increment = 1 / norm_to_episode_len if norm_to_episode_len else 1
        self.histogram = np.zeros(
            (self.action_space.n,), dtype=self.env.observation_space.dtype
        )
        histogram_space = gym.spaces.Box(
            low=np.zeros(
                    self.action_space.n, dtype=float
                ),
            high=np.ones(self.action_space.n,
                            dtype=float)
                    * (1.0 if norm_to_episode_len else np.inf)
                ,
            dtype=float,
        )
        if isinstance(self.observation_space_spec.space, Dict):
            self.observation_space_spec.space = Dict(
            {**self.observation_space_spec.space.spaces, 
             "action_histogram":histogram_space},
            self.observation_space_spec.space.name
            )
            

    def reset(self, *args, **kwargs):
        self.histogram = np.zeros(
            (self.action_space.n,), dtype=self.env.observation_space.dtype
        )
        return super().reset(*args, **kwargs)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        observations=None,
        **kwargs,
    ):
        for a in actions:
            self.histogram[a] += self.increment
        return super().multistep(actions, **kwargs)

    def convert_observation(self, observation):
        if isinstance(observation, dict):
            obs = observation
            obs["action_histogram"] = copy.deepcopy(self.histogram)
        else:
            obs = list(observation)
            obs.append(copy.deepcopy(self.histogram))
        return obs

    def fork(self):
        fork_env =  type(self)(env = self.env.fork(), 
                          norm_to_episode_len = self.norm_to_episode_len)
        fork_env.histogram = copy.deepcopy(self.histogram)
        return fork_env

class AppendAutophaseObservationWrapper(ObservationWrapper):
    # The index of the "TotalInsts" feature of autophase.
    TotalInsts_index = 51
    
    def __init__(self, env: CompilerEnv, normalize=False):
        super().__init__(env)
        self.normalize = normalize
        self.obs_name = 'Autophase'
        if isinstance(self.observation_space_spec.space, Dict):
            self.observation_space_spec.space = Dict(
            {**self.observation_space_spec.space.spaces, 
             "autophase":env.unwrapped.observation.spaces[self.obs_name].space},
            self.observation_space_spec.space.name
            )
    
    def fork(self):
        return type(self)(self.env.fork(), self.normalize)
    
    def convert_observation(self, observation: ObservationType) -> ObservationType:
        autophase_obs = self.env.unwrapped.observation[self.obs_name]
        if self.normalize:
            if autophase_obs[self.TotalInsts_index] <= 0:
                return np.zeros(autophase_obs.shape, dtype=np.float32)
            autophase_obs = np.clip(
                autophase_obs.astype(np.float32) /
                autophase_obs[self.TotalInsts_index], 0, 1
            )
        observation['autophase']=autophase_obs
        return observation
    
class AppendLLMHiddenstatesObservationWrapper(ObservationWrapper):
    
    def __init__(self, env: CompilerEnv, hidden_state_path:str, size:int=768):
        '''
        hidden_state_path should be a path contains multiple .hidden_state file and a metadata.json file.
        metadata.json should contain a dict with benchmark names as key and hiedden state file name as value.
        '''
        super().__init__(env)
        self.hidden_state_path = hidden_state_path
        with open(os.path.join(self.hidden_state_path, "metadata.json"), 'r')as f:
            self.metadata = json.load(f)
        if 'size' in self.metadata:
            size = self.metadata['size']
            print("Hidden state size is set to", size, ", model name:", self.metadata['name'])
        self.size = size
        if isinstance(self.observation_space_spec.space, Dict):
            self.observation_space_spec.space = Dict(
            {**self.observation_space_spec.space.spaces, 
             "llmhidden":gym.spaces.Box(
                    low=np.full(
                            size, -np.inf, dtype=float
                        ),
                    high=np.full(
                            size, np.inf, dtype=float
                        ),
                    dtype=float,
                )},
            self.observation_space_spec.space.name
            )
        self.last_key = None
        
    
    def fork(self):
        return type(self)(self.env.fork(), self.hidden_state_path, self.size)
    
    def convert_observation(self, observation: ObservationType) -> ObservationType:
        # print(self.env.benchmark)
        key = str(self.env.benchmark).split('/')[-1]
        if key not in self.metadata:
            key = str(self.env.benchmark).split('//')[-1].replace('/', '_')
        if key not in self.metadata:
            if key != self.last_key:
                print("[Warning] Benchmark", key, "has no hidden_state file.]")
                self.last_key = key
            observation['llmhidden'] = np.zeros((self.size,))
            return observation
        with open(os.path.join(self.hidden_state_path, self.metadata[key]), 'rb') as f:
            hidden_state = pickle.load(f)
        if type(hidden_state) == torch.Tensor:
            observation['llmhidden'] = hidden_state.detach().cpu().numpy()
        else:
            observation['llmhidden'] = hidden_state
        return observation

class MergeVectorObservationWrapper(ObservationWrapper):
    '''
    Merge all vector-like features if observation is a Dict type.
    '''
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        low_vec = np.array([], dtype=float)
        high_vec = np.array([], dtype=float)
        for key, value in dict(self.observation_space_spec.space.spaces).items():
            if isinstance(value, gym.spaces.Box):
                low_vec = np.concatenate([low_vec, value.low])
                high_vec = np.concatenate([high_vec, value.high])
            elif isinstance(value, gym.spaces.Discrete):
                low_vec = np.concatenate([low_vec, [0]])
                high_vec = np.concatenate([high_vec, [value.n]])
        merge_space = gym.spaces.Box(
            low=low_vec, high=high_vec
        )
        self.observation_space_spec.space = merge_space
            
    def convert_observation(self, observation: ObservationType) -> ObservationType:
        if isinstance(observation, dict):
            return_dict = {}
            merge_vec = []
            for key, value in observation.items():
                if isinstance(value, list) or isinstance(value, np.ndarray):
                    merge_vec+=(list(value))
                elif isinstance(value, int) or isinstance(value, float):
                    merge_vec.append(value)
                else:
                    return_dict[key] = value
            return np.array(merge_vec)
        return observation

class NoLimitActionSpace(ConstrainedCommandline):
    def __init__(self, env: CompilerEnv):
        flags = env.action_space.flags
        super().__init__(env=env, flags=flags)


class AutophaseActionSpace(ConstrainedCommandline):
    """An action space wrapper that limits the action space to that of the
    Autophase paper.

    The actions used in the Autophase work are taken from:

    https://github.com/ucb-bar/autophase/blob/2f2e61ad63b50b5d0e2526c915d54063efdc2b92/gym-hls/gym_hls/envs/getcycle.py#L9

    Note that 4 of the 46 flags are not included. Those are:

        -codegenprepare     Excluded from CompilerGym
            -scalarrepl     Removed from LLVM in https://reviews.llvm.org/D21316
        -scalarrepl-ssa     Removed from LLVM in https://reviews.llvm.org/D21316
             -terminate     Not found in LLVM 10.0.0
    """

    def __init__(self, env: LlvmEnv):
        super().__init__(
            env=env,
            flags=[
                "-adce",
                "-break-crit-edges",
                "-constmerge",
                "-correlated-propagation",
                "-deadargelim",
                "-dse",
                "-early-cse",
                "-functionattrs",
                "-functionattrs",
                "-globaldce",
                "-globalopt",
                "-gvn",
                "-indvars",
                "-inline",
                "-instcombine",
                "-ipsccp",
                "-jump-threading",
                "-lcssa",
                "-licm",
                "-loop-deletion",
                "-loop-idiom",
                "-loop-reduce",
                "-loop-rotate",
                "-loop-simplify",
                "-loop-unroll",
                "-loop-unswitch",
                "-lower-expect",
                "-loweratomic",
                "-lowerinvoke",
                "-lowerswitch",
                "-mem2reg",
                "-memcpyopt",
                "-partial-inliner",
                "-prune-eh",
                "-reassociate",
                "-sccp",
                "-simplifycfg",
                "-sink",
                "-sroa",
                "-strip",
                "-strip-nondebug",
                "-tailcallelim",
            ],
        )

class FunctionActionSpace(ConstrainedCommandline):
    '''
    All 91 FunctionPass CompilerGym supports
    '''
    def __init__(self, env: LlvmEnv):
        super().__init__(
            env=env,
            flags=[
                '-add-discriminators', '-adce', '-aggressive-instcombine', '-alignment-from-assumptions', '-bdce', 
                '-break-crit-edges', '-simplifycfg', '-callsite-splitting', '-consthoist', '-constprop', 
                '-coro-cleanup', '-coro-early', '-coro-elide', '-correlated-propagation', '-dce', '-die', '-dse', 
                '-reg2mem', '-div-rem-pairs', '-early-cse-memssa', '-early-cse', '-ee-instrument', '-flattencfg', 
                '-float2int', '-gvn-hoist', '-gvn', '-guard-widening', '-indvars', '-irce', '-infer-address-spaces', 
                '-inject-tli-mappings', '-instsimplify', '-instcombine', '-instnamer', '-jump-threading', '-lcssa', 
                '-licm', '-libcalls-shrinkwrap', '-load-store-vectorizer', '-loop-data-prefetch', '-loop-deletion', 
                '-loop-distribute', '-loop-fusion', '-loop-guard-widening', '-loop-idiom', '-loop-instsimplify', 
                '-loop-interchange', '-loop-load-elim', '-loop-predication', '-loop-reroll', '-loop-rotate', 
                '-loop-simplifycfg', '-loop-simplify', '-loop-sink', '-loop-reduce', '-loop-unroll-and-jam', 
                '-loop-unroll', '-loop-unswitch', '-loop-vectorize', '-loop-versioning-licm', '-loop-versioning', 
                '-loweratomic', '-lower-constant-intrinsics', '-lower-expect', '-lower-guard-intrinsic', '-lowerinvoke', 
                '-lower-matrix-intrinsics', '-lowerswitch', '-lower-widenable-condition', '-memcpyopt', '-mergeicmps', 
                '-mldst-motion', '-nary-reassociate', '-newgvn', '-pgo-memop-opt', '-partially-inline-libcalls', 
                '-post-inline-ee-instrument', '-mem2reg', '-reassociate', '-redundant-dbg-inst-elim', '-sccp', 
                '-slp-vectorizer', '-sroa', '-scalarizer', '-separate-const-offset-from-gep', '-simple-loop-unswitch', 
                '-sink', '-speculative-execution', '-slsr', '-tailcallelim', '-mergereturn',
            ]
        )

class AutophaseFunctionActionSpace(ConstrainedCommandline):
    '''
    30 FunctionPass in Autophase Passes
    '''
    def __init__(self, env: LlvmEnv):
        super().__init__(
            env=env,
            flags=['-adce', '-break-crit-edges', '-correlated-propagation', '-dse', '-early-cse', '-gvn', '-indvars', 
                   '-instcombine', '-jump-threading', '-lcssa', '-licm', '-loop-deletion', '-loop-idiom', 
                   '-loop-reduce', '-loop-rotate', '-loop-simplify', '-loop-unroll', '-loop-unswitch', 
                   '-lower-expect', '-loweratomic', '-lowerinvoke', '-lowerswitch', '-mem2reg', '-memcpyopt', 
                   '-reassociate', '-sccp', '-simplifycfg', '-sink', '-sroa', '-tailcallelim',
            ]
        )

class SubSequenceSingleStepActionSpace(ConstrainedCommandline):
    """An action space wrapper that limits the action space to that of the
    Autophase paper.

    The actions used in the Autophase work are taken from:

    https://github.com/ucb-bar/autophase/blob/2f2e61ad63b50b5d0e2526c915d54063efdc2b92/gym-hls/gym_hls/envs/getcycle.py#L9

    Note that 4 of the 46 flags are not included. Those are:

        -codegenprepare     Excluded from CompilerGym
            -scalarrepl     Removed from LLVM in https://reviews.llvm.org/D21316
        -scalarrepl-ssa     Removed from LLVM in https://reviews.llvm.org/D21316
             -terminate     Not found in LLVM 10.0.0
    """

    def __init__(self, env: LlvmEnv):
        super().__init__(
            env=env,
            flags=[
                '-instcombine', '-barrier', '-elim-avail-extern', '-rpo-functionattrs', '-globalopt', '-globaldce',
                '-constmerge', '-float2int', '-lower-constant-intrinsics', '-mem2reg', '-deadargelim', '-jump-threading',
                '-correlated-propagation', '-dse', '-tailcallelim', '-loop-simplify', '-lcssa', '-indvars',
                '-loop-idiom', '-loop-deletion', '-loop-unroll', '-mldst-motion', '-gvn', '-memcpyopt', '-sccp', '-bdce',
                '-licm', '-adce', '-strip-dead-prototypes', '-loop-unswitch', '-loop-rotate',
                '-loop-distribute', '-loop-vectorize', '-loop-sink', '-instsimplify', '-div-rem-pairs', '-simplifycfg',
                '-loop-load-elim', '-prune-eh', '-inline', '-functionattrs', '-sroa', '-early-cse', '-lower-expect',
                '-forceattrs', '-inferattrs', '-ipsccp', '-called-value-propagation', '-attributor', '-early-cse-memssa',
                '-speculative-execution', '-reassociate', '-sink', '-break-crit-edges', '-partial-inliner', '-lowerswitch',
                '-strip-nondebug', '-strip', '-loweratomic', '-lowerinvoke', '-loop-reduce'],
        )


class NullActionWrapper(CompilerEnvWrapper):
    '''
    Add a action that has no effect. the index of that action will be env.action_apace.n
    NOTE: Must be inside of a ActionAsStateWrapper
    '''
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
        n_action = self.action_space.n+1
        self.origin_action_space = copy.copy(self.action_space)
        self.action_space = Discrete(n_action, "action-space-with-null-action")
    
    def multistep(self, actions: Iterable[ActionType], observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, reward_spaces: Optional[Iterable[Union[str, Reward]]] = None, observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, rewards: Optional[Iterable[Union[str, Reward]]] = None):
        actions=list(filter(lambda x:x!=self.origin_action_space.n, actions))
        return super().multistep(actions, observation_spaces, reward_spaces, observations, rewards)


class SubsequenceActionSpace(CompilerEnvWrapper):
    def __init__(self, env: CompilerEnv, with_autophase=True):
        super().__init__(env)
        self.with_autophase = with_autophase
        self.build_actions(with_autophase)
        self.super_action_space = copy.copy(self.action_space)
        self.env.action_space = Discrete(
            len(self.action_list), "subseq-actionspace")
    
    def fork(self):
        return type(self)(env=self.env.fork(), 
                          with_autophase = self.with_autophase)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        observations=None,
        **kwargs,
    ):
        real_actions = []
        for a in actions:
            # assert isinstance(a, int)
            assert a >= 0 and a <= len(self.action_list)
            real_actions += map(
                lambda x: self.super_action_space[x], self.action_list[a].split(' '))
        return super().multistep(real_actions, **kwargs)

    def build_actions(self, with_autophase):
        action_list = [
            '-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -constmerge',
            '-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -globaldce -float2int -lower-constant-intrinsics',
            '-instcombine -barrier -elim-avail-extern -rpo-functionattrs -globalopt -mem2reg -deadargelim',
            '-instcombine -jump-threading -correlated-propagation -dse',
            '-instcombine -jump-threading -correlated-propagation',
            '-instcombine',
            '-instcombine -tailcallelim',
            '-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll',
            '-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll -mldst-motion -gvn -memcpyopt -sccp -bdce',
            '-loop-simplify -lcssa -licm -adce',
            '-loop-simplify -lcssa -licm -strip-dead-prototypes -globaldce -constmerge',
            '-loop-simplify -lcssa -licm -strip-dead-prototypes -globaldce -float2int -lower-constant-intrinsics',
            '-loop-simplify -lcssa -licm -loop-unswitch',
            '-loop-simplify -lcssa -loop-rotate -licm -adce',
            '-loop-simplify -lcssa -loop-rotate -licm -strip-dead-prototypes -globaldce -constmerge',
            '-loop-simplify -lcssa -loop-rotate -licm -strip-dead-prototypes -globaldce -float2int -lower-constant-intrinsics',
            '-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch',
            '-loop-simplify -lcssa -loop-rotate -loop-distribute -loop-vectorize',
            '-loop-simplify -lcssa -loop-sink -instsimplify -div-rem-pairs -simplifycfg',
            '-loop-simplify -lcssa -loop-unroll',
            '-loop-simplify -lcssa -loop-unroll -mldst-motion -gvn -memcpyopt -sccp -bdce',
            '-loop-simplify -loop-load-elim',
            '-simplifycfg',
            '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -constmerge -barrier',
            '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -float2int -lower-constant-intrinsics -barrier',
            '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg -deadargelim -barrier',
            '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -dse -barrier',
            '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -barrier',
            '-simplifycfg -reassociate',
            '-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -constmerge',
            '-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -globaldce -float2int -lower-constant-intrinsics',
            '-simplifycfg -sroa -early-cse -lower-expect -forceattrs -inferattrs -ipsccp -called-value-propagation -attributor -globalopt -mem2reg -deadargelim',
            '-simplifycfg -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation -dse',
            '-simplifycfg -sroa -early-cse-memssa -speculative-execution -jump-threading -correlated-propagation',
        ]
        if with_autophase:
            action_list += [
            "-adce",
                "-break-crit-edges",
                "-constmerge",
                "-correlated-propagation",
                "-deadargelim",
                "-dse",
                "-early-cse",
                "-functionattrs",
                "-functionattrs",
                "-globaldce",
                "-globalopt",
                "-gvn",
                "-indvars",
                "-inline",
                "-ipsccp",
                "-jump-threading",
                "-lcssa",
                "-licm",
                "-loop-deletion",
                "-loop-idiom",
                "-loop-reduce",
                "-loop-rotate",
                "-loop-simplify",
                "-loop-unroll",
                "-loop-unswitch",
                "-lower-expect",
                "-loweratomic",
                "-lowerinvoke",
                "-lowerswitch",
                "-mem2reg",
                "-memcpyopt",
                "-partial-inliner",
                "-prune-eh",
                "-reassociate",
                "-sccp",
                "-sink",
                "-sroa",
                "-strip",
                "-strip-nondebug",
                "-tailcallelim",
            ]
        self.action_list = action_list


class SubsequenceActionSpaceTruncated(CompilerEnvWrapper):
    def __init__(self, env: CompilerEnv, with_autophase=True):
        super().__init__(env)
        self.with_autophase = with_autophase
        self.build_actions(with_autophase)
        self.super_action_space = copy.copy(self.action_space)
        self.env.action_space = Discrete(
            len(self.action_list), "subseq-actionspace")
    
    def fork(self):
        return type(self)(env=self.env.fork(), 
                          with_autophase = self.with_autophase)

    def multistep(
        self,
        actions: List[ActionType],
        observation_spaces=None,
        observations=None,
        **kwargs,
    ):
        real_actions = []
        for a in actions:
            # assert isinstance(a, int)
            assert a >= 0 and a <= len(self.action_list)
            real_actions += map(
                lambda x: self.super_action_space[x], self.action_list[a].split(' '))
        return super().multistep(real_actions, **kwargs)

    def build_actions(self, with_autophase):
        action_list = [
                '-instcombine -globalopt -globaldce -constmerge',
                '-instcombine -globalopt -globaldce',
                '-instcombine -globalopt -mem2reg -deadargelim',
                '-instcombine -jump-threading -correlated-propagation -dse',
                '-instcombine -jump-threading -correlated-propagation',
                '-instcombine',
                '-instcombine -tailcallelim',
                '-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll',
                '-loop-simplify -lcssa -indvars -loop-idiom -loop-deletion -loop-unroll -gvn -memcpyopt -sccp',
                '-loop-simplify -lcssa -licm -adce',
                '-loop-simplify -lcssa -licm -globaldce -constmerge',
                '-loop-simplify -lcssa -licm -globaldce',
                '-loop-simplify -lcssa -licm -loop-unswitch',
                '-loop-simplify -lcssa -loop-rotate -licm -adce',
                '-loop-simplify -lcssa -loop-rotate -licm -globaldce -constmerge',
                '-loop-simplify -lcssa -loop-rotate -licm -globaldce',
                '-loop-simplify -lcssa -loop-rotate -licm -loop-unswitch',
                '-loop-simplify -lcssa -loop-rotate',
                '-loop-simplify -lcssa -simplifycfg',
                '-loop-simplify -lcssa -loop-unroll',
                '-loop-simplify -lcssa -loop-unroll -gvn -memcpyopt -sccp',
                '-loop-simplify',
                '-simplifycfg',
                '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -ipsccp -globalopt -globaldce -constmerge',
                '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -ipsccp -globalopt -globaldce',
                '-simplifycfg -prune-eh -inline -functionattrs -sroa -early-cse -lower-expect -ipsccp -globalopt -mem2reg -deadargelim',
                '-simplifycfg -prune-eh -inline -functionattrs -sroa -jump-threading -correlated-propagation -dse',
                '-simplifycfg -prune-eh -inline -functionattrs -sroa -jump-threading -correlated-propagation',
                '-simplifycfg -reassociate',
                '-simplifycfg -sroa -early-cse -lower-expect -ipsccp -globalopt -globaldce -constmerge',
                '-simplifycfg -sroa -early-cse -lower-expect -ipsccp -globalopt -globaldce',
                '-simplifycfg -sroa -early-cse -lower-expect -ipsccp -globalopt -mem2reg -deadargelim',
                '-simplifycfg -sroa -jump-threading -correlated-propagation -dse',
                '-simplifycfg -sroa -jump-threading -correlated-propagation',
        ]
        if with_autophase:
            action_list += [
            "-adce",
                "-break-crit-edges",
                "-constmerge",
                "-correlated-propagation",
                "-deadargelim",
                "-dse",
                "-early-cse",
                "-functionattrs",
                "-functionattrs",
                "-globaldce",
                "-globalopt",
                "-gvn",
                "-indvars",
                "-inline",
                "-ipsccp",
                "-jump-threading",
                "-lcssa",
                "-licm",
                "-loop-deletion",
                "-loop-idiom",
                "-loop-reduce",
                "-loop-rotate",
                "-loop-simplify",
                "-loop-unroll",
                "-loop-unswitch",
                "-lower-expect",
                "-loweratomic",
                "-lowerinvoke",
                "-lowerswitch",
                "-mem2reg",
                "-memcpyopt",
                "-partial-inliner",
                "-prune-eh",
                "-reassociate",
                "-sccp",
                "-sink",
                "-sroa",
                "-strip",
                "-strip-nondebug",
                "-tailcallelim",
            ]
        self.action_list = action_list

class IgnoreException(Exception):
    pass

class MultiStepTimeLimit(CompilerEnvWrapper):
    def __init__(self, env: CompilerEnv, max_step):
        super().__init__(env)
        self.step_cnt = 0
        self.max_step = max_step
    
    def fork(self):
        fork_env = type(self)(env=self.env.fork(), max_step=self.max_step)
        fork_env.step_cnt = self.step_cnt
        return fork_env
    
    def reset(self, *args, **kwargs) -> Optional[ObservationType]:
        self.step_cnt = 0
        return self.env.reset(*args, **kwargs)
    
    def multistep(self, actions: Iterable[ActionType], observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, reward_spaces: Optional[Iterable[Union[str, Reward]]] = None, observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, rewards: Optional[Iterable[Union[str, Reward]]] = None):
        step_res = self.max_step - self.step_cnt
        cut_fg = False
        if len(actions) > step_res:
            actions = actions[:step_res]
            cut_fg = True
        self.step_cnt += len(actions)
        obs, rwd, done, info = super().multistep(actions, observation_spaces, reward_spaces, observations, rewards)
        if self.step_cnt >= self.max_step:
            done = True
            info['TimeLimit.Turcated']=True
            info['TimeLimit.AllExcuted']=cut_fg
        return obs, rwd, done, info

class JustKeepGoingEnv(CompilerEnvWrapper):
    """This wrapper class prevents the step() and close() methods from raising
    an exception.

        Just keep swimming ...
            |\\    o
            | \\    o
        |\\ /  .\\ o
        | |       (
        |/\\     /
            |  /
            |/

    Usage:

        >>> env = compiler_gym.make("llvm-v0")
        >>> env = JustKeepGoingEnv(env)
        # enjoy ...
    """

    def step(self, action, *args, **kwargs):
        
        try:
            if action is None:
                raise IgnoreException("action is None")
            return self.env.step(action, *args, **kwargs)
        except IgnoreException:
            return None, 0, True, {"error_details": "Ignored"}
        except RuntimeTimeoutException:
            return None, 0, True, {"error_details": "Timeout"}
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("step() error: %s", e)
            traceback.print_exc()
            # Return "null" observation / reward.
            return None, 0, True, {"error_details": str(e)}

    def reset(self, *args, **kwargs):
        N_RETRY = 1
        for i in range(N_RETRY): # NOTE: retry here dose not change the benchmark. the retry in llvm-env will change benchmark each retry
            try:
                return super().reset(*args, **kwargs)
            except GraphSizeExceededException as e:
                raise(e)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(f"reset() error, retrying {i+1}/{N_RETRY}...({e})")
                traceback.print_exc()
                self.close()
                continue
                

        # No more retries.
        raise compiler_gym.errors.dataset_errors.BenchmarkInitError

    def close(self):
        try:
            self.env.close()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Ignoring close() error: %s", e)
