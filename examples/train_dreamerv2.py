import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import json
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
    val_replay = wmlib.Replay(logdir / "val_episodes", seed=seed, **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    eval_replay = wmlib.Replay(logdir / "eval_episodes", seed=seed, **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
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

    def per_episode(ep, mode, env):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        if "compilergym" in config.task:
            reduction = ep["reduction"][-1]
            oz, o0, oT = env.get_instcounts()
            if config.reduction_style == "coreset":
                reduction = oz / oT
            def simpl(action):
                return [np.argmax(x) for x in action]
            print(
                f"{mode.title()} episode has \033[33m{float(reduction)}\033[0m reduction, {length} steps and return \033[32m{score}\033[0m, action history {simpl(ep['action'])}, reward history {ep['reward'].tolist()}, O0 {o0}, final {oT}, Oz {oz}."
            )
            logger.scalar(f"{mode}_reduction", float(reduction))
            if np.max(np.abs(ep["reward"])) > 1.0:
                logger.scalar(f"{mode}_max_reward", np.max(np.abs(ep["reward"])))
        else:
            print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        replay = dict(train=train_replay, val=val_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        if mode != 'eval' and mode != 'val':  # NOTE: to aggregate eval results at last
            logger.write()

    print("Create envs.")
    num_eval_envs = 1
    if config.envs_parallel == "none":
        train_envs = [make_env("train", env_id=_) for _ in range(config.envs)]
        if config.enable_val:
            val_envs = [make_env("val", env_id=_) for _ in range(num_eval_envs)]
        eval_envs = [make_env("eval", env_id=_) for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode, env_id: envs.Async(
            functools.partial(make_env, mode, env_id), config.envs_parallel)
        train_envs = [make_async_env("train", _) for _ in range(config.envs)]
        if config.enable_val:
            val_envs = [make_async_env("val", _) for _ in range(num_eval_envs)]
        eval_envs = [make_async_env("eval", _) for _ in range(num_eval_envs)]
    test_env = make_env("test")
    act_space = train_envs[0].act_space
    print("act_space:", act_space)
    obs_space = train_envs[0].obs_space
    train_driver = wmlib.Driver(train_envs, device, action_mask=config.action_mask, coreset_mode=config.coreset_mode)
    train_driver.on_episode(lambda ep, env: per_episode(ep, mode="train", env=env))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(functools.partial(train_replay.add_step, max_return_limit=config.max_return_limit, max_reward_limit=config.max_reward_limit))
    train_driver.on_reset(functools.partial(train_replay.add_step, max_return_limit=config.max_return_limit, max_reward_limit=config.max_reward_limit))
    if config.enable_val:
        val_driver = wmlib.Driver(val_envs, device, action_mask=config.action_mask)
        val_driver.on_episode(lambda ep, env: per_episode(ep, mode="val", env=env))
        # val_driver.on_episode(val_replay.add_episode)
    eval_driver = wmlib.Driver(eval_envs, device, action_mask=config.action_mask, coreset_mode=config.coreset_mode, coreset_enumerate_test=config.coreset_enumerate_test)
    eval_driver.on_episode(lambda ep, env: per_episode(ep, mode="eval", env=env))
    # eval_driver.on_episode(eval_replay.add_episode)
    test_driver = wmlib.Driver([test_env], device, action_mask=config.action_mask, coreset_mode=config.coreset_mode, coreset_enumerate_test=config.coreset_enumerate_test)
    test_driver.on_episode(lambda ep, env: per_test_episode(ep, mode="test", env=env))

    test_results = {}
    def per_test_episode(ep, mode, env):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        reduction = ep["reduction"][-1]
        oz, o0, oT = env.get_instcounts()
        if config.reduction_style == "coreset":
            reduction = oz / oT
        def simpl(action):
            return [np.argmax(x) for x in action]
        with open("log_1.txt", 'a') as f:
            if config.coreset_enumerate_test:
                f.write(
                    json.dumps({'benchmark': env.env.benchmarkName(), 'reduction': float(reduction), 'length': length, 'score': score, 'O0': o0, 'final': oT, 'Oz': oz, 'seq_index':ep["seq_index"]}) + '\n'
                )
            else:
                f.write(
                    json.dumps({'benchmark': env.env.benchmarkName(), 'reduction': float(reduction), 'length': length, 'score': score, 'O0': o0, 'final': oT, 'Oz': oz}) + '\n'
                )
        print(
            f"{mode.title()} episode has \033[33m{float(reduction)}\033[0m reduction, {length} steps and return \033[32m{score}\033[0m, action history {simpl(ep['action'])}, reward history {ep['reward'].tolist()}, O0 {o0}, final {oT}, Oz {oz}."
        )
        if config.coreset_enumerate_test:
            test_results[env.get_instance_id()].append((length, score, float(reduction), ep["seq_index"]))
        else:
            test_results[env.get_instance_id()] = (length, score, float(reduction))

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    random_agent = agents.RandomAgent(act_space)
    if prefill and not config.test_only:
        print(f"Prefill dataset ({prefill} steps).")
        train_driver(random_agent, steps=prefill, episodes=1)
        train_replay._ongoing_eps.clear()
        # eval_driver(random_agent, episodes=1)
        train_driver.reset()
        if config.enable_val:
            val_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    if config.compilergym.programl:
        train_dataset = train_replay.brute_dataset(**config.dataset)
    else:
        train_dataset = iter(train_replay.dataset(**config.dataset))

    def next_batch(iter, fp16=True):
        # casts to fp16 and cuda
        dtype = torch.float16 if wmlib.ENABLE_FP16 and fp16 else torch.float32  # only on cuda
        out = {k: v.to(device=device, dtype=dtype) if k!="programl" else v for k, v in next(iter).items()}
        return out

    # the agent needs 1. init modules 2. go to device 3. set optimizer
    agnt = agents.DreamerV2(config, obs_space, act_space, step)
    agnt = agnt.to(device)
    agnt.init_optimizers()

    if not config.test_only:
        train_agent = wmlib.CarryOverState(agnt.train)
        train_agent(next_batch(train_dataset))  # do initial benchmarking pass
        torch.cuda.empty_cache()  # clear cudnn bechmarking cache
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
        print("Pretrain agent.")
        for _ in range(config.pretrain):
            train_agent(next_batch(train_dataset))
    train_policy = lambda *args, **kwargs: agnt.policy(
        *args, **kwargs, mode="explore" if should_expl(step) else "train")
    eval_policy = lambda *args, **kwargs: agnt.policy(*args, **kwargs, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next_batch(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            # logger.add(agnt.report(next_batch(report_dataset)), prefix="train")
            logger.write(fps=True)
    if not config.test_only:
        train_driver.on_step(train_step)

    eval_count = 0
    best_val_reduction_geom = -1e10
    best_eval_reduction_geom = -1e10
    try:
        while step < config.steps:
            if not config.no_eval:
                logger.write()
                print("Start evaluation.")
                # logger.add(agnt.report(next_batch(eval_dataset)), prefix="eval")
                if not config.test_only:
                    if config.enable_val:
                        val_driver(eval_policy, episodes=config.eval_eps)
                    # eval_driver(eval_policy, episodes=config.eval_eps)

                if config.enable_test and eval_count % config.test_interval == 0:
                    if config.save_all_models:
                        torch.save(agnt.state_dict(), logdir / ("variables_s" + str(int(step)) + ".pt"))
                    # test on all datasets
                    if config.coreset_enumerate_test:
                        test_results = collections.defaultdict(list)
                    else:
                        test_results = {}
                    st = time.time()
                    test_driver(eval_policy, episodes=config.test_eps)
                    ed = time.time()
                    print(f"Test time: {ed-st}")
                    if config.coreset_enumerate_test:
                        test_results = {k:max(v, key=lambda x:x[1]) for k, v in test_results.items()}

                    # log test results
                    # print(test_results)
                    json.dump(test_results, open(logdir / "test_results.json", "w"))
                    print(f"raw test_reduction_geomean", utils.geom_mean([v[2] for k, v in test_results.items()]))
                    print(f"raw test_reduction_max", np.max([v[2] for k, v in test_results.items()]))
                    print(f"raw test_reduction_min", np.min([v[2] for k, v in test_results.items()]))
                    datasets = sorted(list(set([key.split('//')[1].split('-')[0].split('/')[-1] for key in test_results.keys()])))
                    reduction_means = []
                    reduction_geomeans = []
                    for ds in datasets:
                        lengths, returns, reductions = [[v[i] for k, v in test_results.items() if ds+"-" in k] for i in range(3)]
                        logger.scalar(f"test_length_{ds}_mean", np.mean(lengths))
                        logger.scalar(f"test_return_{ds}_mean", np.mean(returns))
                        logger.scalar(f"test_return_{ds}_geomean", utils.geom_mean(returns))
                        logger.scalar(f"test_reduction_{ds}_mean", np.mean(reductions))
                        logger.scalar(f"test_reduction_{ds}_geomean", utils.geom_mean(reductions))
                        reduction_means.append(np.mean(reductions))
                        reduction_geomeans.append(utils.geom_mean(reductions))

                        print(f"test_reduction_{ds}_geomean", utils.geom_mean(reductions))
                        print(f"test_reduction_{ds}_max", np.max(reductions))
                        print(f"test_reduction_{ds}_min", np.min(reductions))
                    logger.scalar(f"test_reduction_mean", np.mean(reduction_means))
                    logger.scalar(f"test_reduction_geomean", utils.geom_mean(reduction_geomeans))
                    print(f"test_reduction_geomean", utils.geom_mean(reduction_geomeans))

                eval_count += 1

                if config.test_only:
                    exit()

            logger.write()  # NOTE: to aggregate eval results
            if not config.no_eval and not config.test_only:
                if config.enable_val and logger._outputs[-1]._stats['val_reduction_geomean'] > best_val_reduction_geom:
                    best_val_reduction_geom = logger._outputs[-1]._stats['val_reduction_geomean']
                    torch.save(agnt.state_dict(), logdir / "variables_best_val.pt")
                # if logger._outputs[-1]._stats['eval_reduction_geomean'] > best_eval_reduction_geom:
                #     best_eval_reduction_geom = logger._outputs[-1]._stats['eval_reduction_geomean']
                #     torch.save(agnt.state_dict(), logdir / "variables_best_eval.pt")

            if config.stop_steps != -1 and step >= config.stop_steps:
                break
            else:
                print("Start training.")
                train_driver(train_policy, steps=config.eval_every)
                torch.save(agnt.state_dict(), logdir / "variables.pt")
    except KeyboardInterrupt:
        print("Keyboard Interrupt - saving agent")
        torch.save(agnt.state_dict(), logdir / "variables.pt")
    except Exception as e:
        print("Training Error:", e)
        torch.save(agnt.state_dict(), logdir / "variables_error.pt")
        raise e
    finally:
        for env in train_envs + eval_envs:
            try:
                env.finish()
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass

    torch.save(agnt.state_dict(), logdir / "variables.pt")


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
