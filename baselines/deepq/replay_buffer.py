import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_samples(self, idxes=[]):
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)



class ReplayBufferPerAction(object):
    def __init__(self, size, num_actions):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [ [] for _ in range(num_actions)]
        self._maxsize = size
        self._next_idx = [0 for _ in range(num_actions)]
        self.num_actions = num_actions

    def __len__(self):
        l = sum([len(strg) for strg in self._storage])
        return l

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx[action] >= len(self._storage[action]):
            self._storage[action].append(data)
        else:
            self._storage[action][self._next_idx[action]] = data
        self._next_idx[action] = (self._next_idx[action] + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        buffers_len = [len(buf) for buf in self._storage]
        buffers_len.insert(0, 0)
        buffers_len = np.cumsum(buffers_len)
        for i in idxes:
            for a in range(self.num_actions):
                if i < buffers_len[a+1] and i >= buffers_len[a]:
                    try:
                        data = self._storage[a][i-buffers_len[a]]
                    except:
                        pass
                    obs_t, action, reward, obs_tp1, done = data
                    obses_t.append(np.array(obs_t, copy=False))
                    actions.append(np.array(action, copy=False))
                    rewards.append(reward)
                    obses_tp1.append(np.array(obs_tp1, copy=False))
                    dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_samples(self, idxes=[]):
        return self._encode_sample(idxes)

    def n_samples_per_action(self, n):
        buffers_len = [len(buf) for buf in self._storage]
        if n > min(buffers_len):
            n = min(buffers_len) #we want to make sure every action gets the same number of samples
        buffers_len.insert(0, 0)
        buffers_len = np.cumsum(buffers_len)
        idxes = []
        for a in range(self.num_actions):
            idxes.extend([random.randint(buffers_len[a], buffers_len[a+1] - 1) for _ in range(n)])
        return self._encode_sample(idxes)

class ReplayBufferPerActionNew(object):
    def __init__(self, size, num_actions):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        # self._storage = [ [] for _ in range(num_actions)]
        # self._maxsize = size
        # self._next_idx = [0 for _ in range(num_actions)]
        self.buffers = [ReplayBuffer(size) for _ in range(num_actions)]
        self.num_actions = num_actions

    def __len__(self):
        l = sum([len(buf) for buf in self.buffers])
        return l

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.buffers[action].add(obs_t, action, reward, obs_tp1, done)

        # if self._next_idx[action] >= len(self._storage[action]):
        #     self._storage[action].append(data)
        # else:
        #     self._storage[action][self._next_idx[action]] = data
        # self._next_idx[action] = (self._next_idx[action] + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        buffers_len = [len(buf) for buf in self.buffers]
        buffers_len.insert(0, 0)
        buffers_len = np.cumsum(buffers_len)
        for i in idxes:
            for a in range(self.num_actions):
                if i < buffers_len[a+1] and i >= buffers_len[a]:
                    data = self.buffers[a].get_samples([i-buffers_len[a]])
                    obs_t, action, reward, obs_tp1, done = data
                    obses_t.append(np.array(obs_t, copy=False))
                    actions.append(np.array(action, copy=False))
                    rewards.append(reward)
                    obses_tp1.append(np.array(obs_tp1, copy=False))
                    dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def get_samples(self, idxes=[]):
        return self._encode_sample(idxes)

    def n_samples_per_action(self, n):
        buffers_len = [len(buf) for buf in self.buffers]
        if n > min(buffers_len):
            n = min(buffers_len) #we want to make sure every action gets the same number of samples
        obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = [], [], [], [], []
        for buf in self.buffers:
            obses_t, actions, rewards, obses_tp1, dones = buf.sample(n)
            obses_t_per_a.append(obses_t)
            actions_per_a.append(actions)
            rewards_per_a.append(rewards)
            obses_tp1_per_a.append(obses_tp1)
            dones_per_a.append(dones)
        return obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a
