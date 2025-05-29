import numpy as np
import torch
import time
import json


class Driver:

    def __init__(self, envs, device, action_mask=False, coreset_mode=False, coreset_enumerate_test=False, **kwargs):
        self._envs = envs
        self._device = device
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self._action_mask = action_mask
        self.coreset_mode = coreset_mode
        self.coreset_enumerate_test = coreset_enumerate_test
        if self.coreset_mode:
            print("Coreset mode is enabled")
            with open("wmlib/envs/coreset_sorted.jsonl", 'r') as f:
                self.coreset = [json.loads(line.strip()) for line in f.readlines()]
                self.current_seq = None
                self.current_seq_index = 0
        self.reset()
        
    def reset_cur_seq(self):
        if self.coreset_enumerate_test:
            self.current_seq = self.coreset[self.current_seq_index].copy()
            self.current_seq_index += 1
            if self.current_seq_index == len(self.coreset):
                self.current_seq_index = 0
        else:
            self.current_seq = self.coreset[np.random.randint(0, len(self.coreset)-1)].copy()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._action_masks = [torch.ones((act_space["action"].shape[0])).to(self._device) for act_space in self._act_spaces]
        self._state = None

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        if self.coreset_enumerate_test:
            episodes *= 50
        while step < steps or episode < episodes:
            obs = {
                i: self._envs[i].reset()
                for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                tran = {k: self._convert(v) for k, v in {**self._obs[i], **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                if self.coreset_mode:
                    if (not self.coreset_enumerate_test) or episode % (episodes//50) == 0:
                        self.reset_cur_seq()
                        print(episode, episodes, self.current_seq_index)
                    else:
                        if self.coreset_enumerate_test:
                            self.current_seq = self.coreset[self.current_seq_index].copy()
                        else:
                            self.current_seq = self.coreset[np.random.randint(0, len(self.coreset)-1)].copy()
                self._eps[i] = [tran]
                self._action_masks[i] = torch.ones((self._act_spaces[i]["action"].shape[0])).to(self._device)

            # obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            obs = {k: torch.from_numpy(np.stack([o[k] for o in self._obs])).float() if k!="programl" else [o[k] for o in self._obs] for k in self._obs[0]}  # convert before sending
            if 'image' in obs and len(obs['image'].shape) == 4:
                obs['image'] = obs['image'].permute(0, 3, 1, 2)
            from .. import ENABLE_FP16
            dtype = torch.float16 if ENABLE_FP16 else torch.float32  # only on cuda
            obs = {k: v.to(device=self._device, dtype=dtype) if k!="programl" else v for k, v in obs.items()}
            action_mask = torch.stack(self._action_masks).to(self._device)

            actions, self._state = policy(obs, self._state, action_mask=action_mask if self._action_mask else None, **self._kwargs)
            actions = [
                {k: np.array(actions[k][i]) for k in actions}
                for i in range(len(self._envs))]
            if self.coreset_mode:
                a = np.zeros((self._act_spaces[0]["action"].shape[0]))
                a[int(self.current_seq[0])] = 1
                actions = [{"action": a.copy()} for _ in range(len(self._envs))]
                self.current_seq = self.current_seq[1:] 
            assert len(actions) == len(self._envs)
            try:
                obs = [e.step(a) for e, a in zip(self._envs, actions)]
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                obs = [
                    {
                        "autophase": self._envs[0].obs_space["autophase"].sample(),
                        "action_histogram": self._envs[0].obs_space["action_histogram"].sample(),
                        "reward": 1e9,
                        "is_first": False,
                        "is_last": True,
                        "is_terminal": True,
                        "reduction": 0.0,
                    }
                    for e, a in zip(self._envs, actions)
                ]
                with open("error.log", "a") as f:
                    for e in self._envs:
                        print("[Warning]", e.get_instance_id(), "env.step() failed, reset envs and continue")
                        f.write(time.ctime() + " " + f"{e.get_instance_id()}\n")
            # obs = [ob() if callable(ob) else ob for ob in obs]
            for i, (act, _ob) in enumerate(zip(actions, obs)):
                ob = _ob() if callable(_ob) else _ob
                if ob["reward"] == 0 - self._envs[i].config.step_penalty:
                    self._action_masks[i][act['action'].argmax()] = 0.0
                else:
                    self._action_masks[i] = torch.ones((self._act_spaces[i]["action"].shape[0])).to(self._device)
                if self.coreset_mode and len(self.current_seq) == 0:
                    ob["is_last"] = True
                self._obs[i] = ob
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1
                if ob['is_last']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    if self.coreset_enumerate_test:
                        ep["seq_index"] = self.current_seq_index - 1
                        if ep["seq_index"] < 0:
                            ep["seq_index"] = len(self.coreset) - 1
                    [fn(ep, env=self._envs[i], **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            # self._obs = obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
