import os
import enum
from functools  import reduce
import compiler_gym.wrappers
import compiler_gym.datasets
import compiler_gym
import compiler_gym.envs
import compiler_gym.envs.llvm.datasets
from compiler_gym.envs.llvm import make_benchmark
from itertools import chain, islice, cycle
from typing import *
from compiler_gym.wrappers import CompilerEnvWrapper
from .wrappers import *
from .llvm_programl_env import *
from .utils import random_with_replacement, random_id_iter, find_file
from random import shuffle
from copy import deepcopy
import json

from .utils import fix_range, POJUrisReordered

'''
Dataset names definition
'''
class DatasetNames(enum.Enum):
    csmith = 0
    github = 1
    tensorflow = 2
    all = 3
    suball = 1003
    suball_extend = 1004
    cbench = 4
    csmith_github_tensorflow = 5
    linux = 6
    poj = 7
    anghabench = 8
    blas = 9
    chstone = 10
    clgen = 11
    mibench = 12
    npb = 13
    opencv = 14
    llvm_stress = 15
    coreset_nvp_test = 16
    coreset_nvp_zeroshot = 17
    cbench22 = 18
    null = 100
    file_dataset = 101
    single_file = 102
    multiple = 103


class BenchmarksManageWrapper(CompilerEnvWrapper):
    '''
    Load the datasets and iterate over benchmarks in it. 
    You can set range , offset and step of the iteration.

    NOTE: 

    To Add datasets:
        1. If you want to use a set of files, follow the steps below:
            * compile all code to .bc using any llvm-based compiler
            * put all .bc file to a folder
            * just use BenchmarksManageWrapper(dataset=DatasetNames.file_dataset, ) 
        2. If you want to add a new dataset of CompilerGym :
            * Add your dataset name to DatasetNames.
            * Modify benchmarkIterFromDataset().
    '''

    def __init__(self,
                 env: CompilerEnv,
                 dataset: DatasetNames = DatasetNames.csmith,
                 start: int = 0,
                 total: int = -1,  # this means use all data
                 offset: int = 0,
                 step: int = 1,
                 if_random: bool = False,
                 if_cycle: bool = True,
                 dataset_path: str = None,
                 metadata_name: str = None,
                 auto_switch_benchmark=True,
                 iterate_seed = 0,
                 programl_max_node=MAX_NODES,
                 programl_max_edge=MAX_EDGES,
                 file_dataset_select=None,
                 ):
        super().__init__(env)
        # Save Params
        self.dataset = dataset
        self.start, self.total, self.offset, self.gap = start, total, offset, step
        self.if_cycle, self.if_random = if_cycle, if_random
        self.auto_switch_benchmark = auto_switch_benchmark
        self.iterate_seed = iterate_seed
        self.programl_max_node, self.programl_max_edge = programl_max_node, programl_max_edge
        if self.dataset == DatasetNames.file_dataset or self.dataset == DatasetNames.single_file:
            assert dataset_path, "file_dataset must specify path to the files."
        self.dataset_path = dataset_path
        self.metadata_name = metadata_name
        self.file_dataset_select = file_dataset_select
        
        self.dataset_name = str(dataset).split(".")[1]
        if self.dataset is DatasetNames.file_dataset or self.dataset is DatasetNames.single_file:
            self.dataset_name = "file::" + self.dataset_path

        # Load Dataset
        if dataset == DatasetNames.null:
            return
        else:
            self.benchmarks, self.benchmark_count = self.get_benchmark_iterator(dataset) 
        if if_cycle and not if_random:
            self.benchmarks = cycle(self.benchmarks)
        dataset_show_name = \
            dataset_path if dataset == DatasetNames.single_file or dataset == DatasetNames.file_dataset\
                else str(dataset).split('.')[-1]
        end = start+total if total>=0 else self.benchmark_count
        import os
        print(
            f"[{self.env.name}] Loaded {self.benchmark_count} benchmarks from {dataset_show_name} st:{start}-tot:{total}-step:{step}-offset:{offset}-pid:{os.getpid()}"
            + (f"-seed:{self.benchmarks.seed}" if hasattr(self.benchmarks, 'seed') else ""))

    def get_benchmark_iterator(self, dataset:DatasetNames):
        if dataset == DatasetNames.null:
            return None, None
        elif dataset == DatasetNames.single_file:  # Only use a single benchmark file.
            return  self.singleBenchmarkFile(
                path=self.dataset_path, 
            )
        # Load multiple benchmarks from a folder and iterate over it.
        elif dataset == DatasetNames.file_dataset:
            return self.benchmarkIterFromFile(
                start=self.start, total=self.total, path=self.dataset_path,
                offset=self.offset, step=self.gap, random=self.if_random,
                name_suffix=".bc", programl_max_node=self.programl_max_node,
                programl_max_edge=self.programl_max_edge, metadata_name = self.metadata_name,
                file_dataset_select=self.file_dataset_select,
            )
        elif dataset == DatasetNames.suball_extend:
            builtin_data_iter, builtin_data_len = self.benchmarkIterFromDataset(
                dataset=DatasetNames.suball, start=self.start, total=self.total, offset=self.offset, step=self.gap, random=self.if_random
            )
            file_data_iter, file_data_len = self.benchmarkIterFromFile(
                start=self.start, total=self.total * 2, path=self.dataset_path,
                offset=self.offset, step=self.gap, random=self.if_random,
                name_suffix=".bc", programl_max_node=self.programl_max_node,
                programl_max_edge=self.programl_max_edge, metadata_name = self.metadata_name,
                file_dataset_select=self.file_dataset_select,
            )
            print("Suball_extend: builtin:", builtin_data_len, "file:", file_data_len)
            return chain(builtin_data_iter, file_data_iter), builtin_data_len + file_data_len
        else:  # Use the datasets of CompilerGym
            return self.benchmarkIterFromDataset(
                dataset=dataset, start=self.start, total=self.total, offset=self.offset, step=self.gap, random=self.if_random
            )
    
    def benchmarkCount(self):
        return self.benchmark_count

    def benchmarkName(self):
        return str(self.benchmark)
    
    def datasetName(self):
        return self.dataset_name

    def reset(self, switch_benchmark=None, **kwargs):
        if switch_benchmark is None:
            switch_benchmark = self.auto_switch_benchmark
        self.step_count = 0
        self.actions_hist = []
        benchmark = next(self.benchmarks) if switch_benchmark else str(
            self.env.benchmark)  # NOTE: This may raise error
        self.total_reward = 0.0
        MAX_RETRY = 128
        # retry if reset failed (This is useful when dataset contains broken benchmarks)
        for i in range(MAX_RETRY):
            try:
                # if switch_benchmark:
                #     obs = self.env.reset(benchmark=benchmark, **kwargs)
                # else:
                #     obs = self.env.reset(**kwargs)
                obs = self.env.reset(benchmark=benchmark, **kwargs)
                if self.env.observation['IrInstructionCountO0'] <= 0:
                    raise Exception(f"Empty benchmark:{benchmark}")
                return obs
            except Exception as e:
                print(
                    f"[Warning] reset() failed, retrying...({i}/{MAX_RETRY}) {benchmark}:{e}")
                # traceback.print_exc()
                benchmark = next(self.benchmarks)

        raise Exception(f"[Error] Failed to reset enviornment!({benchmark})")

    def fork(self):
        '''
        Fork a new Benchmark Manager
        '''
        # If no param given, copy the iterator of current manager.
        fork_env = BenchmarksManageWrapper(
            self.env.fork(),
            dataset=DatasetNames.null,
            start=self.start,
            total=self.total,
            offset=self.offset,
            step=self.gap,
            if_random=self.if_random,
            if_cycle=self.if_cycle,
            dataset_path=self.dataset_path,
            metadata_name=self.metadata_name,
            auto_switch_benchmark=self.auto_switch_benchmark,
            programl_max_node=self.programl_max_node,
            iterate_seed = self.iterate_seed,
            programl_max_edge=self.programl_max_edge, # this has no effect in fork. This param only use in building benchmark iterator, 
            #                                           which will be simply copied when forking.
            file_dataset_select=self.file_dataset_select,
        )
        fork_env.dataset = self.dataset
        fork_env.dataset_name = self.dataset_name
        fork_env.benchmarks = copy.copy(self.benchmarks)
        return fork_env

    def skip_benchmark(self):
        '''
        Just skip a benchmark
        '''
        return next(self.benchmarks)
    
    def switch_benchmark(self):
        '''
        Switch benchmark
        returns True if success, False when benchmarks used up
        '''
        try:
            self.env.reset(benchmark=next(self.benchmarks))
            return True
        except StopIteration as e:
            return False

    def singleBenchmarkFile(
        self,
        path: str,
    ):
        if not os.path.exists(path):
            print(f"[Error] file \'{path}\' not exists.")
            return None, None
        itera = [make_benchmark(path)]
        count = 1
        return iter(itera), count

    def benchmarkIterFromFile(
        self,
        start: int,
        total: int,
        path: str,
        offset: int = 0,
        step:  int = 1,
        random: bool=False,
        name_suffix: str = ".bc",
        metadata_name = None,
        programl_max_node = MAX_NODES,
        programl_max_edge = MAX_EDGES,
        file_dataset_select = None,
    ):
        '''
        Build a Iterator 
        '''
        file_name_list = []
        loaded = False
        
        if file_dataset_select is not None:
            with open(os.path.join(path, "datasets.json"), 'r') as f:
                datasets_dict = json.load(f)
                metadata_name = datasets_dict[file_dataset_select]
        
        if metadata_name is None: metadata_name = "metadata.json"
        metadata_name = find_file(path, metadata_name)
        
        if metadata_name is not None and os.path.exists(metadata_name):
            with open(metadata_name, 'r') as f:
                file_name_list = json.loads(f.read())
                # file_name_list = [os.path.split(i)[-1] for i in file_name_list] # only read filename
                print("Metadata loaded from", metadata_name)
                loaded = True
        else:
            metadata_name = os.path.join(path, "metadata.json")
            for dirpath, dirs, files in os.walk(path):
                # shuffle(files)
                for file in files:
                    prefix, suffix = os.path.splitext(file)
                    if name_suffix is None or name_suffix == suffix:
                        file_name_list.append(file)
                            
            # sort filenames
            def key_func(file_path: str):
                import re
                file_path = os.path.basename(file_path)
                number = re.findall("\d+", str(file_path))
                number = [0]+number
                MOD_V = 998244353
                POW_V = 19260817
                if not number:
                    return -1
                tot = 0
                for i in number:
                    tot = tot * POW_V + int(i)
                return tot*MOD_V + \
                    reduce(lambda x,y: (x*POW_V+y)%MOD_V, bytearray(file_path, encoding='utf-8'))
            file_name_list.sort(key=key_func)
        # random.Random(19260817).shuffle(file_path_list)

        max_node, max_edge = programl_max_node, programl_max_edge

        if max_node and max_edge: # filter programl gragh size
            programl_info_path = os.path.join(path, "programl_info.json")
            if os.path.exists(programl_info_path):
                with open(programl_info_path, "r") as f:
                    info = json.loads(f.read())
                    data: dict = info['data']

                    def filter_func(file_path: str):
                        file_name = os.path.split(file_path)[-1]
                        info = data.get(file_name)
                        if not info:
                            return False
                        return info[0] <= max_node and info[1] <= max_edge
                    file_name_list = list(filter(filter_func, file_name_list))

        benchmark_cnt = len(file_name_list)
        if not loaded: # Save metadata if it is not just loaded from file
            with open(metadata_name, 'w') as f:
                f.write(json.dumps(file_name_list))
                print("Metadata write to", metadata_name)   
        print(f"Found total {benchmark_cnt} benchmarks in {path}")

        class FilePathIter:
            def __init__(self, path_list, path, filter_suffix='.bc'):
                self.path = path
                self.path_list = path_list
                self.current_index = 0
                self.filter_suffix = filter_suffix
                
            def __len__(self):
                return len(self.path_list)
            
            def __getitem__(self, index):
                return self.path_list[index]

            def __iter__(self):
                return self

            def __next__(self):
                benchmark = ''
                while not benchmark.endswith(self.filter_suffix):
                    if self.current_index >= len(self.path_list):
                        raise StopIteration()
                    benchmark = "file://" + os.path.join(self.path, self.path_list[self.current_index])
                    self.current_index += 1
                    if not benchmark.endswith(self.filter_suffix):
                        print(f"[Warning] An invalid item exists in metadata file. Ignored. ({benchmark})")
                return benchmark
        iter = FilePathIter(file_name_list, path)
        start, total = fix_range(start, total, benchmark_cnt)
        iter = islice(iter, start + offset, start + total, step)
        if random:
            iter = random_with_replacement(list(iter), self.iterate_seed)
        return iter, (total - 1 - offset) // step + 1

    def benchmarkIterFromDataset(
        self,
        dataset: DatasetNames = DatasetNames.csmith,
        start: int = 0,
        total: int = -1,
        offset: int = 0,
        step:  int = 1,
        random: bool=False
    ):
        '''
        Build benchmark iterator

        '''
        assert step > 0
        assert not( random and (dataset == DatasetNames.all or dataset == DatasetNames.suball or dataset == DatasetNames.csmith_github_tensorflow or dataset == DatasetNames.suball_extend)), "Can't build a random iterator of multiple generator-typed datasets."
        # build from datasets
        datasets = []
        env = compiler_gym.make("llvm-v0")
        if dataset == DatasetNames.csmith \
                or dataset == DatasetNames.csmith_github_tensorflow:
            datasets.append(env.datasets["generator://csmith-v0"])

        if dataset == DatasetNames.github \
                or dataset == DatasetNames.csmith_github_tensorflow:
            datasets.append(env.datasets["benchmark://github-v0"])

        if dataset == DatasetNames.tensorflow \
                or dataset == DatasetNames.csmith_github_tensorflow:
            datasets.append(env.datasets["benchmark://tensorflow-v0"])
        if dataset == DatasetNames.linux:
            datasets.append(env.datasets["benchmark://linux-v0"])
        if dataset == DatasetNames.poj:
            datasets.append(env.datasets["benchmark://poj104-v1"])
        if dataset == DatasetNames.cbench:
            datasets.append(env.datasets["benchmark://cbench-v1"])
        if dataset == DatasetNames.anghabench:
            datasets.append(env.datasets["benchmark://anghabench-v1"])
        if dataset == DatasetNames.blas:
            datasets.append(env.datasets["benchmark://blas-v0"])    
        if dataset == DatasetNames.chstone:
            datasets.append(env.datasets["benchmark://chstone-v0"]) 
        if dataset == DatasetNames.clgen:
            datasets.append(env.datasets["benchmark://clgen-v0"])    
        if dataset == DatasetNames.mibench:
            datasets.append(env.datasets["benchmark://mibench-v1"])    
        if dataset == DatasetNames.npb:
            datasets.append(env.datasets["benchmark://npb-v0"]) 
        if dataset == DatasetNames.opencv:
            datasets.append(env.datasets["benchmark://opencv-v0"]) 
        if dataset == DatasetNames.llvm_stress:
            datasets.append(env.datasets["generator://llvm-stress-v0"]) 

        if dataset == DatasetNames.all:
            datasets = [env.datasets["benchmark://anghabench-v1"],
                        env.datasets["benchmark://blas-v0"],
                        env.datasets["benchmark://cbench-v1"],
                        env.datasets["benchmark://chstone-v0"],
                        env.datasets["benchmark://clgen-v0"],
                        env.datasets["generator://csmith-v0"],
                        env.datasets["benchmark://github-v0"],
                        env.datasets["benchmark://linux-v0"],
                        env.datasets["generator://llvm-stress-v0"],
                        env.datasets["benchmark://mibench-v1"],
                        env.datasets["benchmark://npb-v0"],
                        env.datasets["benchmark://opencv-v0"],
                        env.datasets["benchmark://poj104-v1"],
                        env.datasets["benchmark://tensorflow-v0"]]
        if dataset == DatasetNames.suball:
            datasets = [env.datasets["benchmark://blas-v0"],
                        env.datasets["benchmark://cbench-v1"],
                        env.datasets["benchmark://chstone-v0"],
                        env.datasets["benchmark://linux-v0"],
                        env.datasets["benchmark://mibench-v1"],
                        env.datasets["benchmark://npb-v0"],
                        env.datasets["benchmark://opencv-v0"],
                        env.datasets["benchmark://tensorflow-v0"]]
        

        if len(datasets) > 0:  # datasetTemplate matches 'if condition' above
            ret_iter = None
            total_in_total = 0
            _offset = offset
            for dataset in datasets:
                dataset.install()
                _start, _total = fix_range(start, total, len(dataset))
                if _offset >= _total:
                    _offset -= _total
                    continue
                total_in_total += (_total - 1 - _offset) // step + 1
                if "generator" in str(dataset) and random:
                    return random_id_iter(str(dataset), _start + _offset, _start + _total, step, self.iterate_seed), total_in_total
                benchmark_iter = dataset.benchmark_uris()
                if "poj" in next(dataset.benchmark_uris()):
                    benchmark_iter = POJUrisReordered(benchmark_iter)
                dataset_iter = islice(
                    benchmark_iter, _start + _offset, _start + _total, step)
                ret_iter = dataset_iter if ret_iter is None else chain(
                    ret_iter, dataset_iter)
                _offset = (step - (_total-_offset) % step) % step
            env.close()
            del env
            if random:
                ret_iter = random_with_replacement(list(ret_iter), self.iterate_seed)
            return ret_iter, total_in_total

        # build from uris
        if len(datasets) == 0:
            if dataset == DatasetNames.cbench22:
                benchmarks = [
                    'benchmark://cbench-v1/adpcm', 
                    'benchmark://cbench-v1/bitcount', 
                    'benchmark://cbench-v1/blowfish', 
                    'benchmark://cbench-v1/bzip2', 
                    'benchmark://cbench-v1/crc32', 
                    'benchmark://cbench-v1/dijkstra', 
                    'benchmark://cbench-v1/gsm', 
                    'benchmark://cbench-v1/ispell', 
                    'benchmark://cbench-v1/jpeg-c', 
                    'benchmark://cbench-v1/jpeg-d', 
                    'benchmark://cbench-v1/lame', 
                    'benchmark://cbench-v1/patricia', 
                    'benchmark://cbench-v1/qsort', 
                    'benchmark://cbench-v1/rijndael', 
                    'benchmark://cbench-v1/sha', 
                    'benchmark://cbench-v1/stringsearch', 
                    'benchmark://cbench-v1/stringsearch2', 
                    'benchmark://cbench-v1/susan', 
                    'benchmark://cbench-v1/tiff2bw', 
                    'benchmark://cbench-v1/tiff2rgba', 
                    'benchmark://cbench-v1/tiffdither', 
                    'benchmark://cbench-v1/tiffmedian'
                    ]
            elif dataset == DatasetNames.coreset_nvp_test:
                from .coreset_nvp_dataset import coreset_nvp_test
                benchmarks = coreset_nvp_test
            elif dataset == DatasetNames.coreset_nvp_zeroshot:
                from .coreset_nvp_dataset import coreset_nvp_zeroshot_test
                benchmarks = coreset_nvp_zeroshot_test
            start, total = fix_range(start, total, len(benchmarks))
            it = islice((env.datasets.benchmark(b)
                        for b in benchmarks), start + offset, start + total, step)
            if random:
                it = random_with_replacement(list(it), self.iterate_seed)
            
            # if random:
            #     ret_iter = random_with_replacement(list(it), self.iterate_seed)
            # env.close()
            # del env
            return it, total

        env.close()
        del env
        print(
            f"[Error] benchmarkIterFromDataset: invalid dataset name {dataset} received. Loaded nothing.")
        return []

class NamedEnvWrapper(CompilerEnvWrapper):
    '''
    Add a name to your enviornment
    '''
    def __init__(self, env: CompilerEnv, name=0):
        super().__init__(env)
        self.name = name
        
class MoreInfoWrapper(CompilerEnvWrapper):
    '''
    Add uri, IrInstructionCount, IrInstructionCountOz, IrInstructionCountO0 to
    Info returned at each step()
    '''
    def __init__(self, env: CompilerEnv):
        super().__init__(env)
    
    def multistep(self, 
                  actions: Iterable[ActionType], 
                  observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, 
                  reward_spaces: Optional[Iterable[Union[str, Reward]]] = None, 
                  observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None, 
                  rewards: Optional[Iterable[Union[str, Reward]]] = None):
        observation, reward, done, info = super().multistep(actions, observation_spaces, reward_spaces, observations, rewards)
        info["uri"] = self.benchmark.uri
        info["IrInstructionCount"] = self.observation["IrInstructionCount"]
        info["IrInstructionCountOz"] = self.observation["IrInstructionCountOz"]
        info["IrInstructionCountO0"] = self.observation["IrInstructionCountO0"]
        return observation, reward, done, info


def BuildBasicalEnviornment(
        observation_space: str = "Autophase",
        reward_space: str = "IrInstructionCount",
        action_space: str = "Autophase",
        dataset_name: DatasetNames = DatasetNames.csmith,
        name:str = "basic_env",
        vectorized_observation=False,
        normalize_feature=True,
        action_histogram: bool = True,
        max_step=45,
        leakiness_factor=0.001,
        count_step: bool = False,  # Add step count to observation
        ir_compare_reward: bool = False,
        reduction_reward: bool = False,
        null_action=False,
        oz_action=False,
        ignore_size_limit=False,
        append_llm_hidden_state=False,
        hidden_state_path=None,
        **kwargs # Use keyword args to pass other env params
    ) -> compiler_gym.CompilerEnv:
    '''
    Build basical enviornment
    This is a factory function, you can select observation, action, reward space and dataset
    and other options to build a basical llvm gym enviornment.
    
    To add more feature to the enviornment, you can write a wrapper, and add some if-else here,
    or wrap envs create by this function somewhere else.
    
    @param observation_space: choice in ["Autophase", "Programl", "InstCount", "AutophasePrograml"]
    @param reward_space: choice in ["IrInstructionCount", "Runtime"]
    @param action_space: choice in ["Autophase", "NoLimit", "Subseq", "SubseqAutophase"]
    @param dataset_name: DatasetNames
    @param name: name of the environment
    
    To set start, step, offset and other benchmark iteration related settings, just use keyword params
    and the params will be sent to proper place.
    
    '''
    # Observation
    if "Programl" in observation_space:
        env = compiler_gym.make("llvm-v0", observation_space="Programl")
    elif observation_space is not None:
        env = compiler_gym.make("llvm-v0",
                                observation_space=observation_space)
    else:
        env = compiler_gym.make("llvm-v0")
    
    # Add name of env
    env = NamedEnvWrapper(env, name)
    
    # Reward
    # TODO: make the logic clearer here
    if reward_space != "Runtime":
        env.reward_space = reward_space
    else:
        env.reward_space = "IrInstructionCount"
        env = RuntimeReward(env)
    if reward_space == "IrInstructionCount":  # TODO: fit ObjectSize
        # env = O0BasedRewardWrapper(env)
        env.reward_space = "IrInstructionCountOz"
    if reward_space == "ObjectTextSizeBytes":  # TODO: fit ObjectSize
        # env = O0BasedRewardWrapper(env)
        env.reward_space = "ObjectTextSizeOz"

    # Action
    if action_space == "Autophase":
        env = AutophaseActionSpace(env)
    if action_space == "NoLimit":
        env = NoLimitActionSpace(env)
    if action_space == "Subseq":
        env = SubSequenceSingleStepActionSpace(env)
        env = SubsequenceActionSpace(env, False)
    if action_space == "SubseqAutophase":
        env = SubSequenceSingleStepActionSpace(env)
        env = SubsequenceActionSpace(env, True)
    if action_space == "SubseqAutophaseTruncated":
        env = SubSequenceSingleStepActionSpace(env)
        env = SubsequenceActionSpaceTruncated(env, True)
    if action_space == "SubseqSingleStep":
        env = SubSequenceSingleStepActionSpace(env)
        
    if null_action:
        env = NullActionWrapper(env)
    if oz_action:
        env = OzOptActionWrapper(env)
    
    # Add Some Features to Observation
    if not "Programl" in observation_space:
        kwargs['programl_max_node'] = 0
        kwargs['programl_max_edge'] = 0
    if "Programl" in observation_space:
        # env = CommandlineWithTerminalAction(env) # TODO: ???
        env = ProgramlWrapper(env, ignore_size_limit=ignore_size_limit, max_node= kwargs['programl_max_node'], max_edge=kwargs['programl_max_edge'])
        env = ObservationToDictionaryWrapper(env, "programl")
        if action_histogram:
            env = AppendActionsHistogram(env, norm_to_episode_len=max_step) # NOTE: normalize action_histogram to 0,1. check if this is necessary.
        if "Autophase" in observation_space:
            env = AppendAutophaseObservationWrapper(env, normalize_feature)
    else:
        if normalize_feature and (observation_space is None or "Autophase" in observation_space):
            env = AutophaseNormalizedFeatures(env)
        env = ObservationToDictionaryWrapper(env, 'autophase')
            
        if action_histogram:
            env = AppendActionsHistogram(env, max_step if (observation_space is None or "Autophase" in observation_space) else 0)
            
        if count_step:
            env = StepCountWrapper(env, max_step)
    if append_llm_hidden_state:
        env = AppendLLMHiddenstatesObservationWrapper(env, hidden_state_path=hidden_state_path)

    # NOTE: this may not be able to use due to some bug in compiler_gym
    # NOTE: abandoned ?
    env = ClampedReward(
        env, leakiness_factor=leakiness_factor)

    # this ignore exceptions in close() and step()
    env = JustKeepGoingEnv(env)
    
    # Limit max steps
    env = MultiStepTimeLimit(
        env, max_step=max_step)

    if ir_compare_reward:
        print("use IrCompare reward")
        env = BinaryRewardWrapper(env)

    if reduction_reward:
        print("use Reduction reward")
        env = ReductionRewardWrapper(env)
        
    # Dataset Management
    env = BenchmarksManageWrapper(env, **{**{"dataset":dataset_name}, **kwargs})
        
    # Add More step info for logging
    env = MoreInfoWrapper(env)

    if vectorized_observation:
        env = MergeVectorObservationWrapper(env)

    return env