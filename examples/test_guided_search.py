import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import json
import math
import time

try:
    import rich.traceback
    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorboard
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml
import torch
import random
from collections import defaultdict

import wmlib
import wmlib.envs as envs
import wmlib.agents as agents
import wmlib.utils as utils


def main():

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent.parent / "configs" / "dreamerv2.yaml").read_text()
    )
    parsed, remaining = utils.Flags(configs=["defaults"]).parse(known_only=True)
    config = utils.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = utils.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    load_logdir = pathlib.Path(config.load_logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir", logdir)

    utils.snapshot_src(".", logdir / "src", ".gitignore")

    message = "No GPU found. To actually train on CPU remove this assert."
    assert torch.cuda.is_available(), message  # FIXME
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        wmlib.ENABLE_FP16 = True  # enable fp16 here since only cuda can use fp16
        print("setting fp16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device != "cpu":
        torch.set_num_threads(1)

    # reproducibility
    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # no apparent impact on speed
    torch.backends.cudnn.benchmark = True  # faster, increases memory though.., no impact on seed

    train_replay = wmlib.Replay(logdir / "train_episodes", seed=seed, **config.replay)
    step = utils.Counter(train_replay.stats["total_steps"])
    outputs = [
        utils.TerminalOutput(),
        utils.JSONLOutput(logdir),
        utils.TensorBoardOutputPytorch(logdir),
    ]
    logger = utils.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = utils.Every(config.train_every)
    should_log = utils.Every(config.log_every)
    should_expl = utils.Until(config.expl_until // config.action_repeat)

    # save experiment used config
    with open(logdir / "used_config.yaml", "w") as f:
        f.write("## command line input:\n## " + " ".join(sys.argv) + "\n##########\n\n")
        yaml.dump(config, f)

    def make_env(mode, env_id=0, step_limit=None):
        suite, task = config.task.split("_", 1)
        if suite == "compilergym":
            env = envs.CompilerGym(
                config.seed + env_id,
                envs.CompilerGymConfig()\
                    .set_dataset(task if mode != "test" else config.test_dataset)\
                    .update(**config.compilergym)\
                    .update(**{"max_step":step_limit} if step_limit is not None else {}),
                mode="test" if mode =="eval" else mode,
            )
            env = envs.OneHotAction(env)
        else:
            raise NotImplementedError(suite)
        env = envs.TimeLimit(env, config.time_limit if step_limit is None else step_limit)
        return env

    test_env = make_env("test")
    long_test_env = make_env("test", step_limit=200)
    obs_space = test_env.obs_space
    act_space = test_env.act_space
    # the agent needs 1. init modules 2. go to device 3. set optimizer
    agnt = agents.DreamerV2(config, obs_space, act_space, step)
    agnt = agnt.to(device)
    agnt.init_optimizers()
    
    if (load_logdir / "variables.pt").exists():
        print("Load agent.")
        # agnt.load_state_dict(torch.load(load_logdir / "variables.pt"))
        if (load_logdir / "variables_best_val.pt").exists():
            agnt.load_state_dict(torch.load(load_logdir / "variables_best_val.pt"))
        elif (load_logdir / "variables_best_eval.pt").exists():
            agnt.load_state_dict(torch.load(load_logdir / "variables_best_eval.pt"))
        else:
            agnt.load_state_dict(torch.load(load_logdir / "variables.pt"))
    elif (logdir / "variables.pt").exists():
        print("Load agent.")
        agnt.load_state_dict(torch.load(logdir / "variables.pt"))
    else:
        assert False
    
    # Start Eval
    
    max_cum_rwd_for_each_benchmark = defaultdict(list)
    
    LOAD = None
    # LOAD = "/workspace/dengchaoyi/codezero-aaai/CodeZero/max_cum_rwd_for_each_benchmark-2732.json"
    
    WALL_TIME = 60
    # P_RANDOM = 0.00
    P_RANDOM = 0.05
    
    from wmlib import ENABLE_FP16
    dtype = torch.float16 if ENABLE_FP16 else torch.float32  # only on cuda
    with torch.cuda.amp.autocast(enabled=ENABLE_FP16):
        with torch.no_grad():
           if LOAD is None:
                for episode in range(int(config.test_eps)):
                    obs = [test_env.reset(),]
                    print("Search:", test_env.get_instance_id())
                    st = time.time()
                    max_cum_rwd = 0
                    max_seq = []
                    max_reduction = 0
                    while time.time()-st < WALL_TIME:
                        print("  Reset")
                        obs = [test_env.reset(switch_benchmark=False)]
                        state = None
                        cum_rwd = 0
                        seq = []
                        while not obs[0]["is_last"] and time.time()-st < WALL_TIME:
                            obs = {k: torch.from_numpy(np.stack([o[k] for o in obs])).float() if k!="programl" else [o[k] for o in obs] for k in obs[0]}
                            obs = {k: v.to(device=device, dtype=dtype) if k!="programl" else v for k, v in obs.items()}
                            actions, state = agnt.policy(obs, state, action_mask=None, mode="explore")
                            actions = [
                                {k: np.array(actions[k][0]) for k in actions},
                            ]
                            if np.random.rand() < P_RANDOM:
                                print("  --random")
                                a = np.random.choice(act_space["action"].n)
                                action = np.zeros(act_space["action"].n)
                                action[a]=1
                                actions = [{"action":action},]
                            seq.append(int(actions[0]["action"].argmax()))
                            try:
                                obs = [e.step(a) for e, a in zip([test_env, ], actions)]
                            except Exception as e:
                                print(e)
                                break
                            cum_rwd += obs[0]["reward"]
                            if cum_rwd > max_cum_rwd:
                                max_cum_rwd = cum_rwd
                                max_seq = seq.copy()
                                max_reduction = obs[0]["reduction"]
                                print("  New Max:", max_cum_rwd, max_seq, max_reduction)
                            # break
                        # break
                    
                    # extend 
                    if False:
                        print("EXTEND")
                        obs = [long_test_env.reset(switch_benchmark=True)]
                        for act in max_seq:
                            obs = [long_test_env.step({"action":torch.nn.functional.one_hot(torch.tensor(act), num_classes=act_space['action'].n).float().numpy()})]
                        stp = 0
                        while not obs[0]["is_last"]:
                            stp += 1
                            obs = {k: torch.from_numpy(np.stack([o[k] for o in obs])).float() if k!="programl" else [o[k] for o in obs] for k in obs[0]}
                            obs = {k: v.to(device=device, dtype=dtype) if k!="programl" else v for k, v in obs.items()}
                            actions, state = agnt.policy(obs, state, action_mask=None, mode="explore")
                            actions = [
                                {k: np.array(actions[k][0]) for k in actions},
                            ]
                            seq.append(int(actions[0]["action"].argmax()))
                            obs = [e.step(a) for e, a in zip([long_test_env, ], actions)]
                            cum_rwd += obs[0]["reward"]
                            if cum_rwd > max_cum_rwd:
                                max_cum_rwd = cum_rwd
                                max_seq = seq.copy()
                                max_reduction = obs[0]["reduction"]
                                print("  New Max:", max_cum_rwd, max_seq, max_reduction)
                        print(stp)
                    max_cum_rwd_for_each_benchmark[test_env.get_instance_id()]=((max_cum_rwd, max_seq, max_reduction, time.time()-st))

    if LOAD is None:
        with open(f"max_cum_rwd_for_each_benchmark-{np.random.randint(0,9999):04d}-{P_RANDOM:.3f}-{config.actor.temp:.2f}.json", "w") as f:
            json.dump(max_cum_rwd_for_each_benchmark, f)
        print("Search Done.")
    else:
        with open(LOAD, "r") as f:
            max_cum_rwd_for_each_benchmark = json.load(f)
        print("Search Loaded.")
    
        
    
    test_results = max_cum_rwd_for_each_benchmark
    print(f"raw test_reduction_geomean", utils.geom_mean([v[2] for k, v in test_results.items()]))
    print(f"raw test_reduction_max", np.max([v[2] for k, v in test_results.items()]))
    print(f"raw test_reduction_min", np.min([v[2] for k, v in test_results.items()]))
    print(f"raw test_time mean", np.mean([v[3] for k, v in test_results.items()]))
    datasets = sorted(list(set([key[12:].split('-')[0] for key in test_results.keys()])))
    reduction_means = []
    reduction_geomeans = []
    for ds in datasets:
        lengths, returns, reductions = [[v[i] for k, v in test_results.items() if ds in k] for i in range(3)]
        reduction_means.append(np.mean(reductions))
        reduction_geomeans.append(utils.geom_mean(reductions))

        print(f"test_reduction_{ds}_geomean", utils.geom_mean(reductions))
        print(f"test_reduction_{ds}_max", np.max(reductions))
        print(f"test_reduction_{ds}_min", np.min(reductions))
        print(f"test_time_{ds} mean", np.mean([v[3] for k, v in test_results.items() if ds in k]))
        


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
