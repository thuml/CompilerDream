
import collections
import datetime
import io
import pathlib
import uuid
import collections
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, IterableDataset


class Replay:

    def __init__(
        self,
        directory,
        load_directory=None,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
        dreamsmooth=0.0,
        seed=0,
    ):
        self._directory = pathlib.Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._ongoing = ongoing
        self._minlen = minlen
        self._maxlen = maxlen
        self._prioritize_ends = prioritize_ends
        self._dreamsmooth = dreamsmooth
        self._random = np.random.RandomState(seed)

        if load_directory is None:
            load_directory = self._directory
        else:
            load_directory = pathlib.Path(load_directory).expanduser()

        # filename -> key -> value_sequence
        self._complete_eps = load_episodes(load_directory, capacity, minlen)
        # worker -> key -> value_sequence
        self._ongoing_eps = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        self._total_episodes, self._total_steps = count_episodes(directory)
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())

    @property
    def stats(self):
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "loaded_steps": self._loaded_steps,
            "loaded_episodes": self._loaded_episodes,
        }

    def add_step(self, transition, max_return_limit=None, max_reward_limit=None, worker=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            if (max_return_limit is None or abs(np.sum(episode["reward"])) <= max_return_limit) and (max_reward_limit is None or np.max(np.abs(episode["reward"])) <= max_reward_limit):
                self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode, env=None):
        length = eplen(episode)
        if length < self._minlen:
            print(f"Skipping short episode of length {length}.")
            return
        if self._dreamsmooth:
            alpha = self._dreamsmooth
            for t in range(1, len(episode["reward"])):
                episode["reward"][t] = alpha * episode["reward"][t - 1] + (1 - alpha) * episode["reward"][t]
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: convert(value, iden=(key == "programl")) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["action"])
                if len(sequence["action"]) < 1:
                    sequence = self._sample_sequence()
            programl = chunk["programl"] if "programl" in chunk else None
            chunk = {k: np.concatenate(v) for k, v in chunk.items() if k != "programl"}
            if programl is not None:
                tmp = []
                for v in programl:
                    tmp += v
                chunk["programl"] = np.array(tmp)
            # from T,H,W,C to T,C,H,W
            if "image" in chunk and len(chunk["image"].shape) == 4:
                chunk["image"] = chunk["image"].transpose(0, 3, 1, 2)
            yield chunk

    def _sample_sequence(self):
        episodes = list(self._complete_eps.values())
        if self._ongoing:
            episodes += [
                x for x in self._ongoing_eps.values() if eplen(x) >= self._minlen
            ]
        episode = self._random.choice(episodes)
        total = len(episode["reward"])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        # Randomize length to avoid all chunks ending at the same time in case the
        # episodes are all of the same length.
        length -= np.random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        sequence = {
            k: convert(v[index: index + length], iden=(k == "programl"))
            for k, v in episode.items()
            if not k.startswith("log_")
        }
        sequence["is_first"] = np.zeros(len(sequence["reward"]), bool)
        # NOTE: for sequence, is_first=True at start point
        #       but between chunks, this may not hold and need to carry state
        sequence["is_first"][0] = True
        if self._maxlen:
            assert self._minlen <= len(sequence["reward"]) <= self._maxlen
        return sequence

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            # Relying on Python preserving the insertion order of dicts.
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]

    def dataset(self, batch, length, pin_memory=True, **kwargs):
        generator = lambda: self._generate_chunks(length)

        class ReplayDataset(IterableDataset):
            def __iter__(self):
                return generator()

        dataset = ReplayDataset()
        # TODO we cant use workers, since it forks, and the _episodes variable will be outdated
        # TODO: speed up without DataLoader
        dataset = DataLoader(
            dataset,
            batch,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=seed_worker,
            **kwargs
        )  # ,num_workers=2)
        return dataset

    def brute_dataset(self, batch, length, pin_memory=True, **kwargs):
        generator = self._generate_chunks(length)
        while True:
            batchs = []
            for i in range(batch):
                batchs.append(next(generator))
            final_batch = {}
            for k in batchs[0]:
                if k != "programl":
                    final_batch[k] = torch.from_numpy(np.stack([x[k] for x in batchs], 0))
                else:
                    final_batch[k] = [x[k] for x in batchs]
            yield final_batch


class ReplayWithoutAction(Replay):

    def __init__(
        self,
        directory,
        load_directory=None,
        capacity=0,
        ongoing=False,
        minlen=1,
        maxlen=0,
        prioritize_ends=False,
        seed=0,
    ):
        super().__init__(
            directory=directory,
            load_directory=load_directory,
            capacity=capacity,
            ongoing=ongoing,
            minlen=minlen,
            maxlen=maxlen,
            prioritize_ends=prioritize_ends,
            seed=seed,
        )

    def _generate_chunks(self, length):
        sequence = self._sample_sequence()
        sequence = {
            k: v
            for k, v in sequence.items()
            if k in ["is_first", "is_last", "is_terminal", "image"]
        }
        sequence["action"] = np.zeros((sequence["image"].shape[0], 1), dtype=np.float32)

        while True:
            chunk = collections.defaultdict(list)
            added = 0
            while added < length:
                needed = length - added
                adding = {k: v[:needed] for k, v in sequence.items()}
                sequence = {k: v[needed:] for k, v in sequence.items()}
                for key, value in adding.items():
                    chunk[key].append(value)
                added += len(adding["image"])
                if len(sequence["image"]) < 1:
                    sequence = self._sample_sequence()
                    sequence = {
                        k: v
                        for k, v in sequence.items()
                        if k in ["is_first", "is_last", "is_terminal", "image"]
                    }
                    sequence["action"] = np.zeros(
                        (sequence["image"].shape[0], 1), dtype=np.float32
                    )

            chunk = {k: np.concatenate(v) for k, v in chunk.items()}
            # from T,H,W,C to T,C,H,W
            if len(chunk["image"].shape) == 4:
                chunk["image"] = chunk["image"].transpose(0, 3, 1, 2)
            yield chunk


def count_episodes(directory):
    filenames = list(directory.glob("*.npz"))
    num_episodes = len(filenames)
    num_steps = sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)
    return num_episodes, num_steps


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f"{timestamp}-{identifier}-{length}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open("wb") as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None, minlen=1):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob("*.npz"))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split("-")[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open("rb") as f:
                episode = np.load(f, allow_pickle=True)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f"Could not load episode {str(filename)}: {e}")
            continue
        episodes[str(filename)] = episode
    return episodes


def convert(value, iden=False):
    if iden:
        return value
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


def eplen(episode):
    return len(episode["reward"]) - 1


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
