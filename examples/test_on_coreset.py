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

    def make_env(mode, env_id=0):
        suite, task = config.task.split("_", 1)
        if suite == "compilergym":
            env = envs.CompilerGym(
                config.seed + env_id,
                envs.CompilerGymConfig()\
                    .set_dataset(task if mode != "test" else config.test_dataset)\
                    .update(**config.compilergym),
                mode="test" if mode =="eval" else mode,
            )
            env = envs.OneHotAction(env)
        else:
            raise NotImplementedError(suite)
        env = envs.TimeLimit(env, config.time_limit)
        return env

    test_env = make_env("test")
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
    
    # load coreset
    with open("wmlib/envs/coreset_sorted.jsonl", 'r') as f:
        coreset = [json.loads(line.strip()) for line in f.readlines()]
        current_seq = None
        current_seq_index = 0
    
    class CoresetPolicy:
        def __init__(self, seq, act_space):
            self.current_seq = seq.copy()
            self.act_space = act_space
        
        def __call__(self, *args, **kwargs):
            assert self.current_seq
            a = np.zeros((act_space["action"].shape[0]))
            a[int(self.current_seq[0])] = 1
            self.current_seq = self.current_seq[1:]
            return {"action":a}, None
    
    def imagine(wm:agents.WorldModel, policy, start, init_weight, is_terminal, horizon, dtype, device, sample=True):
        start["feat"] = wm.rssm.get_feat(start)
        start["action"] = torch.zeros((1, act_space["action"].shape[0])).to(dtype=dtype, device=device)
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = torch.from_numpy(np.stack([policy(seq["feat"][-1].detach())[0]["action"]])).to(dtype=dtype, device=device)
            state = wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action, sample=sample)
            feat = wm.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if "discount" in wm.heads:
            disc = wm.heads["discount"](seq["feat"]).mean
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - is_terminal.to(dtype=disc.dtype, device=disc.device)
                true_first *= wm.config.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = wm.config.discount * torch.ones(seq["feat"].shape[:-1]).to(seq["feat"].device)
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        
        seq["weight"] = torch.cumprod(
            # torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
            torch.cat([init_weight.unsqueeze(0).to(disc.device), disc[:-1]], 0), 0
        )
        return seq
    
    max_cum_rwd_for_each_benchmark = defaultdict(list)
    
    RANDOM = False
    LOAD = None
    from wmlib import ENABLE_FP16
    dtype = torch.float16 if ENABLE_FP16 else torch.float32  # only on cuda
    with torch.cuda.amp.autocast(enabled=ENABLE_FP16):
        with torch.no_grad():
            for K in range(50):
                if (LOAD is not None) or RANDOM:
                    continue
                for episode in range(int(config.test_eps)):
                    obs = [test_env.reset(),]
                    current_seq = coreset[current_seq_index].copy()
                    obs = {k: torch.from_numpy(np.stack([o[k] for o in obs])).float() if k!="programl" else [o[k] for o in obs] for k in obs[0]}
                    if 'image' in obs and len(obs['image'].shape) == 4:
                        obs['image'] = obs['image'].permute(0, 3, 1, 2)
                    
                    obs = {k: v.to(device=device, dtype=dtype) if k!="programl" else v for k, v in obs.items()}
                    
                    latent = agnt.wm.rssm.initial(len(obs["reward"]), device)
                    embed = agnt.wm.get_embed(agnt.wm.preprocess(obs), eval=True)
                    action = torch.zeros((len(obs["reward"]),) + agnt.act_space.shape).to(device)
                    latent, _ = agnt.wm.rssm.obs_step(
                        latent, action, embed, obs["is_first"], sample = not config.eval_state_mean)
                    
                    policy = CoresetPolicy(current_seq, act_space)
                    seq = imagine(agnt.wm, policy, latent, torch.tensor([1.0]), torch.tensor([False]), len(current_seq), dtype, device, sample = not config.eval_state_mean)
                    feat = seq['feat']
                    
                    reconstruct = defaultdict(list)
                    for name, head in agnt.wm.heads.items():
                        for fea in feat:
                            reconstruct[name].append(head(fea))
                    
                    img_autophases = [i["autophase"].mean.detach().cpu().numpy() for i in reconstruct["decoder"]]
                    img_rewards = [i.mean.detach().cpu().item() for i in reconstruct["reward"][1:]]
                    img_reductions = [i.detach().cpu().item() for i in reconstruct["reduction"]]
                    
                    rel_autophase, rel_rewards = [], []
                    for step in current_seq:
                        a = np.zeros((act_space["action"].shape[0]))
                        a[step] = 1
                        rel_obs = test_env.step({"action":a})
                        rel_autophase.append(rel_obs["autophase"])
                        rel_rewards.append(rel_obs["reward"])
                    benchmark = test_env.get_instance_id()
                    # print("Autophase MSE:", [(i-j).pow(2).mean().item() for i, j in zip(img_autophases, rel_autophase)])
                    print("Reward MSE:", [(i-j).pow(2).mean().item() for i, j in zip(img_rewards, rel_rewards)])
                    with open(logdir / "test_detail.jsonl", "a") as f:
                        f.write(json.dumps(
                            {
                                'benchmark':benchmark,
                                'actions':current_seq,
                                'img_rewards':img_rewards,
                                'rel_rewards':rel_rewards,
                                'img_autophases':img_autophases,
                                'rel_autophase':rel_autophase,
                            }
                        ) + '\n')
                    for i in range(len(img_rewards)+1):
                        cum_reward = sum([0.0]+img_rewards[:i])
                        action_seq = current_seq[:i]
                        max_cum_rwd_for_each_benchmark[benchmark].append((cum_reward, img_rewards[:i], action_seq))
                current_seq_index = 0 if current_seq_index+1 == len(coreset) else current_seq_index+1
    if RANDOM:
        for episode in range(int(config.test_eps)):
            test_env.reset()
            benchmark = test_env.get_instance_id()
            print(benchmark)
            for seq in coreset:
                for i in range(len(seq)+1):
                    cum_reward = [0.0]+[np.random.random()]
                    max_cum_rwd_for_each_benchmark[benchmark].append((sum(cum_reward), cum_reward, seq[:i]))
    else:
        if LOAD is None:
            with open(f"max_cum_rwd_for_each_benchmark-4683.json", "w") as f:
                json.dump(max_cum_rwd_for_each_benchmark, f)
            print("Imaginary Search Done.")
        else:
            with open(LOAD, "r") as f:
                max_cum_rwd_for_each_benchmark = json.load(f)
            print("Imaginary Search Loaded.")
    
    
    # 
    results = {}
    for benchmark, data in max_cum_rwd_for_each_benchmark.items():
        data.sort(key=lambda x: x[0], reverse=True) # sort with cum_reward
        # data.sort(key=lambda x: x[3], reverse=True) # sort with img_reduction
        test_env.reset(switch_benchmark=True)
        for max_rwd, img_rewards, max_seq in data[:45]:
            test_env.reset(switch_benchmark=False)
            cur_benchmark = test_env.get_instance_id()
            assert benchmark == cur_benchmark
            rel_cum_reward = []
            for action in max_seq:
                a = np.zeros((act_space["action"].shape[0]))
                a[action] = 1
                obs = test_env.step({"action":a})
                rel_cum_reward.append(obs["reward"])
            reduction = test_env.get_reduction()
            print(benchmark, "reduction:", reduction, "img_cum_rwd:", max_rwd, "rel_cum_reward:", sum(rel_cum_reward), "cum_rwd_MSE=", (max_rwd-sum(rel_cum_reward))**2)
            if cur_benchmark not in results or results[cur_benchmark][2] < reduction:
                results[cur_benchmark] = (len(max_seq), sum(rel_cum_reward), reduction, rel_cum_reward, img_rewards)
    
    for benchmark, (length, rel_cum_reward, reduction, rel_rewards, img_rewards) in results.items():
        print(benchmark, "length:", length, "rel_cum_reward:", rel_cum_reward, "reduction:", reduction)
        print("   ", "rel_rewards:", rel_rewards)
        print("   ", "img_rewards:", img_rewards)
        print("")
    
    test_results = results
    print(f"raw test_reduction_geomean", utils.geom_mean([v[2] for k, v in test_results.items()]))
    print(f"raw test_reduction_max", np.max([v[2] for k, v in test_results.items()]))
    print(f"raw test_reduction_min", np.min([v[2] for k, v in test_results.items()]))
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
        


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
