import os, json

import gym
import numpy as np

try:
    from .compilergym_bed.llvm_env import DatasetNames, BuildBasicalEnviornment
except:
    from compilergym_bed.llvm_env import DatasetNames, BuildBasicalEnviornment

from .compilergym_bed.utils import symlog, symexp


def leaky(x, leakiness=0.001):
    return np.sign(x) * (max(abs(x) - 1, 0) * leakiness + min(abs(x), 1))


class CompilerGymConfig:
    def __init__(self):
        # Game config
        self.dataset = 'file_dataset'
        self.mix_weight = [1, 1, 1]
        self.train_size = None
        self.max_step = 45
        self.terminatable = True
        self.has_shuffled = False

        # Reward
        self.baseline_thre = 1.0
        self.leakiness = 1.0
        self.step_penalty = 0.0
        self.rew_space = "IrInstructionCount"

        # Action space
        self.act_space = "Autophase"
        self.oz_act = False
        self.null_act = False

        # Obs space
        self.abs_inst_obs = False
        self.rel_inst_obs = False
        self.instc_list_obs = False
        self.append_llm_hidden_state = False
        self.programl = False

        # Testing config
        self.start_test = 0
        self.test_total_episodes = 50
        self.test_on_gpu = False

    def set_dataset(self, dataset):
        self.dataset = dataset
        return self

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert k in self.__dict__, f"Unknown config {k}"
            self.__dict__[k] = v
        return self

    @property
    def instcount_obs(self):
        return self.abs_inst_obs or self.rel_inst_obs or self.instc_list_obs


def get_dataset_args(dataset_name, mode='train', has_shuffled=False, seed=0):
    if dataset_name.startswith('file_dataset'):
        file_dataset_name = dataset_name[13:]
        dataset_name = 'file_dataset'
    elif dataset_name.startswith('suball_extend'):
        file_dataset_name = 'suball_extend'
    else:
        file_dataset_name = None
    dataset = DatasetNames.__dict__[dataset_name]

    if dataset_name == 'file_dataset' and file_dataset_name == 'codecontest':
        if mode == 'train':
            start, total, step = 0, -1, 1
            metadata_name = "metadata_small_train.json"
        elif mode == 'val':
            start, total, step = 0, 50, 1
            metadata_name = "metadata_small_valid.json"
        elif mode == 'test':
            start, total, step = 0, 100, 1
            metadata_name = "metadata_small_test.json"
        else:
            raise NotImplementedError
    elif dataset_name == 'coreset_nvp_test' or dataset_name == 'coreset_nvp_zeroshot':
        start, total, step = 0, -1, 1
        metadata_name = None
    else:
        if mode == 'train':
            start, total, step = 100, -1, 1
        elif mode == 'val':
            start, total, step = 50, 50, 1
        elif mode == 'test':
            start, total, step = 0, 50, 1
        else:
            raise NotImplementedError
        metadata_name = None

    return dataset, file_dataset_name, start, total, step, metadata_name

def mix_dataset(weight:list, has_shuffled:bool,  seed:int):
    assert len(weight) == 3 and sum(weight) != 0 and min(weight) >= 0
    weight = np.array(weight)/np.sum(weight)
    weight_str = "-".join([f"{i:.3f}" for i in weight])
    main_path = "./data"
    metadata_paths = ["./formai", "./angha", "./codecontest"]
    file_paths = [[],[],[]]
    for k, metadata_path in enumerate(metadata_paths):
        metadata = json.load(open(os.path.join(main_path, metadata_path, "metadata.json" if "codecontest" not in metadata_path else "metadata_small_train.json")))
        file_paths[k] = [os.path.join(metadata_path, i) for i in metadata]
    from math import floor
    dataset_size = min(*[floor(len(file_paths[i])/(weight[i]+1e-4)) for i in range(3)])
    rdm = np.random.RandomState(seed)
    mix_metadata = sum([rdm.choice(file_paths[i], floor(dataset_size*weight[i]), replace=False).tolist() for i in range(3)], start=[])
    if has_shuffled:
        rdm.shuffle(mix_metadata)
    dataset_size = len(mix_metadata)
    print("mix dataset size:", dataset_size)
    mix_meta_name = "metadat-"+weight_str+".json"
    mix_mata_path = os.path.join(main_path, mix_meta_name)
    with open(mix_mata_path, "w") as f:
        json.dump(mix_metadata, f)
    return main_path, mix_mata_path, mix_meta_name

class CompilerGym:

    def __init__(self, seed=None, config: CompilerGymConfig = None, mode='train'):
        if seed is None:
            seed = config.seed
        
        test_mode = (mode != 'train')
        dataset, file_dataset_name, start, total, step, metadata_name = get_dataset_args(
            config.dataset, mode=mode, has_shuffled=config.has_shuffled, seed=seed
        )
        if mode == 'train' and config.train_size is not None:
            total = config.train_size

        if file_dataset_name == 'formai':
            dataset_path = "./data/formai"
        elif file_dataset_name == 'objc':
            dataset_path = "./data/objc/"
        elif file_dataset_name == 'angha':
            dataset_path = "./data/angha"
        elif file_dataset_name == 'codecontest':
            dataset_path = "./data/codecontest"
        elif file_dataset_name == 'mix':
            dataset_path, _, metadata_name = mix_dataset(config.mix_weight, config.has_shuffled, seed)
        elif file_dataset_name == 'suball_extend':
            dataset_path = "./data/fromai_and_angha_testset"
        else:
            dataset_path = None

        self.env = BuildBasicalEnviornment(**{
            'dataset': dataset,
            'start': start,
            'total': total,
            'step': step,
            'metadata_name': metadata_name,

            'observation_space': "Autophase" if not config.programl else "AutophasePrograml",
            'reward_space': config.rew_space,

            'action_space': config.act_space,
            'null_action': config.null_act,
            'oz_action': config.oz_act,

            'max_step': config.max_step,
            'append_llm_hidden_state': config.append_llm_hidden_state,

            'dataset_path': dataset_path,
            'hidden_state_path': None,

            'iterate_seed': seed,
            'auto_switch_benchmark': False,
            'leakiness_factor': config.leakiness,
            'reduction_reward': False,

            'programl_max_node': 1000000 if test_mode else 10000,
            'programl_max_edge': 1000000 if test_mode else 10000,
            'if_random': not test_mode and not config.has_shuffled,
        })

        self.env.seed(seed)
        self.mode = mode
        self.config = config

        # space shape
        self._autophase_shape = 56
        self._action_space_shape = {
            "Autophase": 42,
            "SubseqAutophase": 42 + 32,
            "SubseqAutophaseTruncated": 42 + 32,
            "SubseqSingleStep": 61,
            "NoLimit": 124,
        }[config.act_space] + int(config.null_act) + int(config.oz_act)
        self._instcount_shape = int(
            self.config.abs_inst_obs) * 3 + int(self.config.rel_inst_obs) * 1 + int(self.config.instc_list_obs) * 69
        self._oz_act = self._action_space_shape - 1 if config.oz_act else -1
        self._null_act = self._action_space_shape - 1 - \
            int(config.oz_act) if config.null_act else -1

        # benchmark state
        self._instance_id = None
        self._instcount_baseline = None
        self._instcount_init = None
        self._instcount_last = None

    @property
    def obs_space(self):
        spaces = {
            "autophase": gym.spaces.Box(-np.inf, np.inf, (self._autophase_shape,), dtype=np.float32),
            "action_histogram": gym.spaces.Box(-np.inf, np.inf, (self._action_space_shape,), dtype=np.float32),
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "reduction": gym.spaces.Box(-np.inf, np.inf, (), dtype=bool),
        }
        if self.config.rel_inst_obs or self.config.abs_inst_obs or self.config.instc_list_obs:
            spaces["instcount"] = gym.spaces.Box(
                -np.inf, np.inf, (self._instcount_shape,), dtype=np.float32)
        return spaces

    @property
    def act_space(self):
        action = self.env.action_space
        return {"action": action}

    def get_instcount_obs(self, observation):
        instcount_obs = []
        if self.config.abs_inst_obs:
            instcount_obs += [
                symlog(observation["IrInstructionCountO0"] / 1000),
                symlog(observation["IrInstructionCountOz"] / 1000),
                symlog(observation["IrInstructionCount"] / 1000),
            ]
        if self.config.rel_inst_obs:
            instcount_obs.append(symlog(
                observation["IrInstructionCountOz"] / observation["IrInstructionCount"] - 1.0))
        if self.config.instc_list_obs:
            instcount_obs = observation["InstCountNorm"]
        return np.array(instcount_obs)

    def step(self, action):
        observation, reward, done, _ = self.env.step(action["action"])

        # compute reward
        if self.config.rew_space == "IrInstructionCount":
            current_instcount = self.env.observation["IrInstructionCount"]
        elif self.config.rew_space == "ObjectTextSizeBytes":
            current_instcount = self.env.observation["ObjectTextSizeBytes"]
        else:
            raise NotImplementedError

        reward = (self._instcount_last - current_instcount) / \
            max(self._instcount_init - self._instcount_baseline,
                self.config.baseline_thre)
        reward = leaky(reward, self.config.leakiness)
        reward = reward - self.config.step_penalty

        # update benchmark state
        self._instcount_last = current_instcount

        # process observation
        if self.config.terminatable and action["action"] == self._null_act:
            done = True
        obs = {
            "autophase": observation["autophase"],
            "action_histogram": observation["action_histogram"],
            "reward": reward,
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
            "reduction": self.get_reduction()
        }
        if self.config.programl:
            obs["programl"] = observation["programl"]
        if self.config.instcount_obs:
            obs["instcount"] = self.get_instcount_obs(self.env.observation)
        return obs

    def reset(self, switch_benchmark=True):
        observation = self.env.reset(switch_benchmark)

        # update benchmark state
        if self.config.rew_space == "IrInstructionCount":
            self._instcount_baseline = self.env.observation["IrInstructionCountOz"]
            self._instcount_init = self.env.observation["IrInstructionCountO0"]
            self._instcount_last = self.env.observation["IrInstructionCount"]
        elif self.config.rew_space == "ObjectTextSizeBytes":
            self._instcount_baseline = self.env.observation["ObjectTextSizeOz"]
            self._instcount_init = self.env.observation["ObjectTextSizeO0"]
            self._instcount_last = self.env.observation["ObjectTextSizeBytes"]
        else:
            raise NotImplementedError
        self._instance_id = str(self.env.benchmark.uri)
        if switch_benchmark:
            print(f"{self.mode}benchmark", self.env.benchmark.uri)

        # process observation
        obs = {
            "autophase": observation["autophase"],
            "action_histogram": observation["action_histogram"],
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "reduction": self.get_reduction()
        }
        if self.config.programl:
            obs["programl"] = observation["programl"]
        if self.config.instcount_obs:
            obs["instcount"] = self.get_instcount_obs(self.env.observation)
        return obs

    def close(self):
        self.env.close()

    def get_reduction(self):
        return self.env.observation["IrInstructionCountOz"] / max(1, self.env.observation["IrInstructionCount"])

    def get_instance_id(self):
        return self._instance_id

    def get_instcounts(self):
        return self._instcount_baseline, self._instcount_init, self._instcount_last
