import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func, build_q_func_and_features

#additions
from tqdm import tqdm
import cvxpy as cvx
from datetime import datetime
from random import shuffle
debug_flag = False
structred_learning = False
first_time = True

class BLRParams(object):
    def __init__(self):
        self.sigma = 0.001 #0.001 W prior variance
        self.sigma_n = 1 # noise variance
        self.alpha = .01 # forgetting factor
        if debug_flag:
            self.update_w = 1 # multiplied by update target frequency
            self.sample_w = 1000
        else:
            self.sample_w = 10000
            self.update_w = 10 # multiplied by update target frequency
        self.batch_size = 1000000# batch size to do blr from
        self.gamma = 0.99 #dqn gamma
        self.feat_dim = 128 #256
        self.first_time = True
        self.no_prior = True
        self.a0 = 6
        self.b0 = 6


def information_transfer_new(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, batch_size, num_actions, feat_dim, sdp_ops):
    d = [[] for i in range(num_actions)]
    phi = [[] for i in range(num_actions)]
    n = [0 for i in range(num_actions)]
    print("transforming information")
    from datetime import datetime
    fmt = '%Y-%m-%d %H:%M:%S'
    d1 = datetime.now()
    information_transfer_new.calls += 1

    phiphiT_inv = np.zeros_like(phiphiT)
    print("phiphiT inv")
    for i in range(num_actions):
        phiphiT_inv[i] = np.linalg.pinv(phiphiT[i])
        # print("inverse norm {}".format(i))
        # print(np.linalg.norm(phiphiT_inv[i]))

    idxes = [i for i in range(len(replay_buffer))]
    shuffle(idxes)
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes[:batch_size])
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
    mini_batch_size = 32*num_actions
    for j in tqdm(range((batch_size // mini_batch_size)+1)):
        # obs_t, action, reward, obs_tp1, done = obses_t[j], actions[j], rewards[j], obses_tp1[j], dones[j]
        start_idx = j*mini_batch_size
        end_idx = (j+1)*mini_batch_size if (j+1)*mini_batch_size < len(replay_buffer) else -1
        obs_t = obses_t[start_idx:end_idx]
        action = actions[start_idx:end_idx]
        for k in range(num_actions):
            if obs_t[action == k].shape[0] < 1:
                continue
            nk = obs_t[action == k].shape[0]
            pseudo_count_k, outer_k = sdp_ops(obs_t[action == k], obs_t[action == k], np.tile(phiphiT_inv[k],(nk,1,1)))
            outer_k = [np.array(p) for p in outer_k.tolist()]
            pseudo_count_k = pseudo_count_k.tolist()
            d[k].extend(pseudo_count_k)
            phi[k].extend(outer_k)
            n[k] += nk

    precisions_return = []
    cov = []
    prior = 0.00001 * np.eye(feat_dim)
    print("solving optimization")
    for a in range(num_actions):
        print("for action {}".format(a))
        if d[a] != []:
            X = cvx.Variable((feat_dim, feat_dim), PSD=True)
            # Form objective.
            obj = cvx.Minimize(sum([(cvx.trace(X * phi[a][i]) - np.squeeze(d[a][i])) ** 2 for i in range(len(d[a]))]))
            prob = cvx.Problem(obj)
            prob.solve(solver=cvx.SUPER_SCS)
            # prob.solve()
            if X.value is None:
                print("failed - cvxpy couldn't solve sdp")
                precisions_return.append(np.linalg.inv(prior))
                cov.append(prior)
                information_transfer_new.failure_num += 1
            else:
                precisions_return.append(np.linalg.inv(X.value + prior))
                cov.append(X.value + prior)
                information_transfer_new.success_num += 1
                information_transfer_new.last_success = information_transfer_new.calls
        else:
            print("failed - no samples")
            precisions_return.append(np.linalg.inv(prior))
            cov.append(prior)
    d2 = datetime.now()
    print("total time for information transfer:")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    print("current call:")
    print(information_transfer_new.calls)
    print('total success:')
    print(information_transfer_new.success_num)
    print('total failure:')
    print(information_transfer_new.failure_num)
    print('last success on:')
    print(information_transfer_new.last_success)
    return precisions_return, cov
information_transfer_new.success_num = 0
information_transfer_new.failure_num = 0
information_transfer_new.calls = 0
information_transfer_new.last_success = 0

def information_transfer_linear(phiphiT, dqn_feat, old_feat, num_actions, feat_dim,replay_buffer):
    idxes = [i for i in range(len(replay_buffer))]
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes)
    obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = [], [], [], [], []
    idxes_per_a = []
    d1 = datetime.now()
    for i in range(num_actions):
        obses_t_per_a.append(obses_t[actions == i])
        actions_per_a.append(actions[actions == i])
        rewards_per_a.append(rewards[actions == i])
        obses_tp1_per_a.append(obses_tp1[actions == i])
        dones_per_a.append(dones[actions == i])
        ix = [j for j in range(obses_t[actions == i].shape[0])]
        shuffle(ix)
        idxes_per_a.append(ix)
    phiphiT0 = np.zeros_like(phiphiT)
    for i in range(num_actions):
        phi_m = None
        xi_m = None
        mi = obses_t_per_a[i].shape[0]
        M = min([1000, mi])
        if M < feat_dim:
            phiphiT0[i] = 1/0.001 * np.eye(feat_dim)
            continue
        for m in range(M):
            phi_t = old_feat(obses_t_per_a[i][idxes_per_a[i][m]][None]).T
            xi_t = dqn_feat(obses_t_per_a[i][idxes_per_a[i][m]][None]).T
            if phi_m is None:
                phi_m = phi_t
            else:
                phi_m = np.concatenate([phi_m,phi_t],axis=-1)
            if xi_m is None:
                xi_m = xi_t
            else:
                xi_m = np.concatenate([xi_m, xi_t],axis=-1)
        phiphiT0[i] = (xi_m @ np.linalg.pinv(phi_m)) @ phiphiT[i] @ (np.linalg.pinv(phi_m).T @ xi_m.T) + 1/0.001 * np.eye(feat_dim)

    d2 = datetime.now()
    print("total time for linear prior")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    return phiphiT0

def information_transfer_single(phiphiT, dqn_feat, target_dqn_feat,
                                replay_buffer, batch_size, num_actions, feat_dim,
                                sdp_ops, old_networks, blr_counter, blr_idxes=[]):
    blr_params = BLRParams()
    phiphiT_inv = np.zeros_like(phiphiT)
    for i in range(num_actions):
        phiphiT_inv[i] = np.linalg.pinv(phiphiT[i])
    d = []
    phi = []
    d1 = datetime.now()
    information_transfer_single.calls += 1
    idxes = [i for i in range(len(replay_buffer))]
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes)
    obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = [], [], [], [], []
    idxes_per_a = []
    for i in range(num_actions):
        obses_t_per_a.append(obses_t[actions == i])
        actions_per_a.append(actions[actions == i])
        rewards_per_a.append(rewards[actions == i])
        obses_tp1_per_a.append(obses_tp1[actions == i])
        dones_per_a.append(dones[actions == i])
        ix = [j for j in range(obses_t[actions == i].shape[0])]
        shuffle(ix)
        idxes_per_a.append(ix)
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)

    """
    mini_batch_size = 32*num_actions
    for j in tqdm(range((batch_size // mini_batch_size)+1)):
        # obs_t, action, reward, obs_tp1, done = obses_t[j], actions[j], rewards[j], obses_tp1[j], dones[j]
        start_idx = j*mini_batch_size
        end_idx = (j+1)*mini_batch_size if (j+1)*mini_batch_size < len(replay_buffer) else -1
        obs_t = obses_t[start_idx:end_idx]
        action = actions[start_idx:end_idx]
        for k in range(num_actions):
            if obs_t[action == k].shape[0] < 1:
                continue
            nk = obs_t[action == k].shape[0]

            pseudo_count_k, outer_k = sdp_ops(obs_t[action == k], obs_t[action == k], np.tile(phiphiT_inv[k],(nk,1,1)))

            outer_k = [np.array(p) for p in outer_k.tolist()]
            pseudo_count_k = pseudo_count_k.tolist()
            d.extend(pseudo_count_k)
            phi.extend(outer_k)
            pass
    """

    for k in range(num_actions):
        if obses_t_per_a[k].shape[0] < 1:
            continue
        nk = obses_t_per_a[k].shape[0]
        nk = min([50,nk])
        idxes = idxes_per_a[k][:nk]
        pseudo_count_k, outer_k = sdp_ops(obses_t_per_a[k][idxes], obses_t_per_a[k][idxes], np.tile(phiphiT_inv[k],(nk,1,1)))


        outer_k = [np.array(p) for p in outer_k.tolist()]
        pseudo_count_k = pseudo_count_k.tolist()
        d.extend(pseudo_count_k)
        phi.extend(outer_k)
        pass

    print("len d")
    print(len(d))
    prior = 0.00001 * np.eye(feat_dim)
    precisions_return = np.linalg.inv(prior)
    cov = prior
    print("solving optimization")
    if d != []:
        X = cvx.Variable((feat_dim, feat_dim), PSD=True)
        # Form objective.
        obj = cvx.Minimize(sum([(cvx.trace(X * phi[i]) - np.squeeze(d[i])) ** 2 for i in range(len(d))]))
        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.SUPER_SCS)
        if X.value is None:
            print("failed - cvxpy couldn't solve sdp")
            information_transfer_single.failure_num += 1
        else:
            information_transfer_single.success_num += 1
            information_transfer_single.last_success = information_transfer_single.calls
            # when solving for phiphiT
            # precisions_return = X.value #+ np.linalg.inv(prior)
            # cov = np.linalg.inv(X.value + np.linalg.inv(prior))
            #when solving for phiphiT_inv
            cov = X.value + prior
            precisions_return = np.linalg.inv(X.value + prior)
            # print('phiphiT0 norm')
            # print(np.linalg.norm(X.value + np.linalg.inv(prior)))
            # print('cov norm')
            # print(np.linalg.norm(cov))
            # for g, e in zip(d, phi):
            #         est = np.trace(np.matmul(X.value,e))
            #         mse = sum((g-e)**2)
            #         print("est:")
            #         print(est)
            #         print("ground truth:")
            #         print(g)
            #         print("mse:")
            #         print(mse)

    else:
        print("failed - no samples")
    d2 = datetime.now()
    print("total time for information transfer:")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    print("current call:")
    print(information_transfer_single.calls)
    print('total success:')
    print(information_transfer_single.success_num)
    print('total failure:')
    print(information_transfer_single.failure_num)
    print('last success on:')
    print(information_transfer_single.last_success)
    return precisions_return, cov
information_transfer_single.success_num = 0
information_transfer_single.failure_num = 0
information_transfer_single.calls = 0
information_transfer_single.last_success = 0

def information_transfer(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, batch_size, num_actions, feat_dim):
    d = [[] for i in range(num_actions)]
    phi = [[] for i in range(num_actions)]
    n = [0 for i in range(num_actions)]
    print("transforming information")
    from datetime import datetime
    fmt = '%Y-%m-%d %H:%M:%S'
    d1 = datetime.now()
    for j in range(batch_size):
        obs_t, action, reward, obs_tp1, done = replay_buffer.sample(1)

        # confidence scores for old data
        c = target_dqn_feat(obs_t).reshape((feat_dim, 1))

        d[int(action)].append(np.dot(np.dot(c.T, phiphiT[int(action)]), c))

        # new data correlations
        c = dqn_feat(obs_t).reshape((feat_dim, 1))
        phi[int(action)].append(np.outer(c, c))
        n[int(action)] += 1
    print(n,sum(n))
    precisions_return = []
    cov = []
    prior = (1/0.001) * np.eye(feat_dim)
    print("solving optimization")
    for a in range(num_actions):
        print("for action {}".format(a))
        if d[a] != []:
            X = cvx.Variable((feat_dim, feat_dim), PSD=True)
            # Form objective.
            obj = cvx.Minimize(sum([(cvx.trace(X * phi[a][i]) - np.squeeze(d[a][i])) ** 2 for i in range(len(d[a]))]))
            prob = cvx.Problem(obj)
            prob.solve()
            if X.value is None:
                print("failed - cvxpy couldn't solve for action {}".format(a))
                precisions_return.append(np.linalg.inv(prior))
                cov.append(prior)
            else:
                precisions_return.append(np.linalg.inv(X.value + prior))
                cov.append(X.value + prior)
        else:
            print("failed - no samples for this action - {}".format(a))
            precisions_return.append(np.linalg.inv(prior))
            cov.append(prior)
    d2 = datetime.now()
    print("total time for information transfer:")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    return precisions_return, cov

def BayesRegression(phiphiT, phiY, replay_buffer, dqn_feat, target_dqn_feat, num_actions, blr_param,
                    w_mu, w_cov, last_layer_weights, prior="sdp", blr_ops=None, sdp_ops=None, old_networks=None, blr_counter=None,old_feat=None):
    # dqn_ and target_dqn_ are feature extractors for the dqn and target dqn respectivley
    # in the neural linear settings target are the old features and dqn are the new features
    # last_layer_weights are the target last layer weights
    feat_dim = blr_param.feat_dim

    n_samples = blr_param.batch_size if len(replay_buffer) > blr_param.batch_size else len(replay_buffer)

    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(n_samples)
    idxes = [i for i in range(len(replay_buffer))]
    shuffle(idxes)

    phiphiT0 = None
    if prior == "no prior":
        print("no prior")
        phiY *= (1-blr_param.alpha)*0
        phiphiT *= (1-blr_param.alpha)*0
        for i in range(num_actions):
            phiphiT[i] = (1/blr_param.sigma)*np.eye(feat_dim)
    elif prior == "linear":
        print("linear")
        phiphiT0 = information_transfer_linear(phiphiT, dqn_feat, old_feat,num_actions,feat_dim, replay_buffer)
        phiphiT *= (1-blr_param.alpha)*0
        phiY *= (1-blr_param.alpha)*0
    elif prior == "decay":
        print("simple prior")
        phiY *= (1-blr_param.alpha)
        phiphiT *= (1-blr_param.alpha)
    elif prior == "last layer":
        print("last layer weights only prior")
        phiY *= (1-blr_param.alpha)*0
        phiphiT *= (1-blr_param.alpha)*0
        for i in range(num_actions):
            phiphiT[i] = (1/blr_param.sigma)*np.eye(feat_dim)
    elif prior == "sdp":
        print("SDP prior")
        phiphiT0, cov0 = information_transfer_new(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, 300*num_actions      , num_actions, feat_dim, sdp_ops)
        for j in range(num_actions):
            print("old phiphiT[{}] new features norm:".format(j))
            print(np.linalg.norm(phiphiT0[j]))
            print("old cov0[{}] new features norm:".format(j))
            print(np.linalg.norm(cov0[j]))
        phiphiT *= (1-blr_param.alpha)*0
        phiY *= (1-blr_param.alpha)*0
    elif prior == "single sdp":
        print("single SDP prior")
        # phiphiT0 = 1/blr_param.sigma * np.eye(feat_dim)
        # phiphiT0 = None
        # if np.any(phiphiT != np.zeros_like(phiphiT)):
        print("using 600")
        phiphiT0, cov0 = information_transfer_single(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, 600      , num_actions, feat_dim, sdp_ops, old_networks, blr_counter)
        phiphiT *= (1-blr_param.alpha)*0
        phiY *= (1-blr_param.alpha)*0

    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes[:n_samples])
    n = np.zeros(num_actions)
    action_rewards = [0. for _ in range(num_actions)]

    mini_batch_size = 32*num_actions
    for j in tqdm(range((n_samples // mini_batch_size)+1)):
        # obs_t, action, reward, obs_tp1, done = obses_t[j], actions[j], rewards[j], obses_tp1[j], dones[j]
        start_idx = j*mini_batch_size
        end_idx = (j+1)*mini_batch_size if (j+1)*mini_batch_size < len(replay_buffer) else -1
        obs_t = obses_t[start_idx:end_idx]
        action = actions[start_idx:end_idx]
        reward = rewards[start_idx:end_idx]
        obs_tp1 = obses_tp1[start_idx:end_idx]
        done = dones[start_idx:end_idx]
        for k in range(num_actions):
            if obs_t[action == k].shape[0] < 1:
                continue
            phiphiTk, phiYk = blr_ops(obs_t[action == k], action[action == k], reward[action == k], obs_tp1[action == k], done[action == k])
            phiphiT[k] += phiphiTk
            phiY[k] += phiYk
            n[k] += obs_t[action == k].shape[0]
            action_rewards[k] += sum(reward[action == k])
    print(n, np.sum(n))
    return phiphiT, phiY, phiphiT0, last_layer_weights

    # old BayesReg func
    # for i in range(num_actions):
    #     if prior == "sdp":# and phiphiT0 is not None:
    #         if i == 0:
    #             print("regular sdp")
    #         if phiphiT0 is None:
    #             inv = np.linalg.inv(phiphiT[i])
    #             print("inv {}".format(i))
    #             print(np.linalg.norm(inv))
    #             w_mu[i] = np.array(np.dot(inv,phiY[i]))
    #         else:
    #             inv = np.linalg.inv(phiphiT[i] + phiphiT0[i])
    #             w_mu[i] = np.array(np.dot(inv,(phiY[i] + np.dot(phiphiT0[i], last_layer_weights[:,i]))))
    #             phiphiT[i] += phiphiT0[i]
    #
    #     elif prior == "single sdp":
    #         # shared phiphiT0
    #         if i == 0:
    #             print("single sdp")
    #         if blr_counter == 0:
    #             inv = np.linalg.inv(phiphiT[i])
    #             w_mu[i] = np.array(np.dot(inv,(phiY[i])))# + np.dot(phiphiT0, last_layer_weights[:,i]))))
    #         else:
    #             print("phiphiT[{}] phiphiT0".format([i]))
    #             print([np.linalg.norm(phiphiT[i]), np.linalg.norm(phiphiT0)])
    #             inv = np.linalg.inv(phiphiT[i] + phiphiT0)
    #             w_mu[i] = np.array(np.dot(inv,(phiY[i]/blr_param.sigma_n + np.dot(phiphiT0, last_layer_weights[:,i]))))
    #             phiphiT[i] += phiphiT0
    #     elif prior == "last layer":
    #         if i == 0:
    #             print("last layer weights only prior")
    #         inv = np.linalg.inv(phiphiT[i]/blr_param.sigma_n + 1/blr_param.sigma * np.eye(feat_dim))
    #         w_mu[i] = np.array(np.dot(inv,(phiY[i]/blr_param.sigma_n + (1/blr_param.sigma)*last_layer_weights[:,i])))
    #     else:
    #         if i == 0:
    #             print("prior: {}".format(prior))
    #         inv = np.linalg.inv(phiphiT[i]/blr_param.sigma_n + 1/blr_param.sigma * np.eye(feat_dim))
    #         w_mu[i] = np.array(np.dot(inv,phiY[i]))/blr_param.sigma_n
    #
    #     w_cov[i] = blr_param.sigma*inv
    #     cov_norms[i] = np.linalg.norm(w_cov[i])
    # print("covariance matrices norms:")
    # print(cov_norms)
    # # print("reward gathered for actions")
    # # print(action_rewards)
    # # for i in range(num_actions):
    # #     print("phiphiT[{}] norm".format(i))
    # #     print(np.linalg.norm(phiphiT[i]))
    # return phiphiT, phiY, w_mu, w_cov

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          thompson=True,
          prior="no prior",
          **network_kwargs
            ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    blr_params = BLRParams()

    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    # q_func = build_q_func(network, **network_kwargs)
    q_func = build_q_func_and_features(network, hiddens=[blr_params.feat_dim], **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug, blr_additions = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),#tf.train.RMSPropOptimizer(learning_rate=lr,momentum=0.95),#
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        thompson=thompson,
        double_q=thompson
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    num_actions = env.action_space.n
    if thompson:
        # Create parameters for Bayesian Regression
        feat_dim = blr_additions['feat_dim']
        num_models = 5
        print("num models is: {}".format(num_models))
        w_sample = np.random.normal(loc=0, scale=blr_params.sigma, size=(num_actions, num_models, feat_dim))
        w_mu = np.zeros((num_actions, feat_dim))
        w_cov = np.zeros((num_actions, feat_dim,feat_dim))
        for i in range(num_actions):
            w_cov[i] = blr_params.sigma*np.eye(feat_dim)

        phiphiT = np.zeros((num_actions,feat_dim,feat_dim))
        for i in range(num_actions):
            phiphiT[i] = (1/blr_params.sigma)*np.eye(feat_dim)
        phiY = np.zeros((num_actions, feat_dim))
        model_idx = np.random.randint(0,num_models,size=num_actions)
        w_norms = [np.linalg.norm(w_sample[i,model_idx[i]]) for i in range(num_actions)]
        blr_ops = blr_additions['blr_ops']
        blr_ops_old = blr_additions['blr_ops_old']
        last_layer_weights = None
        phiphiT0 = None

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    if thompson:
        blr_additions['update_old']()
        if blr_additions['old_networks'] is not None:
            for key in blr_additions['old_networks'].keys():
                blr_additions['old_networks'][key]["update"]()

    episode_rewards = [0.0]
    # episode_Q_estimates = [0.0]
    unclipped_episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        actions_hist = [0 for _ in range(num_actions)]
        actions_hist_total = [0 for _ in range(num_actions)]
        last_layer_weights_decaying_average = None
        blr_counter = 0
        action_buffers_size = 128
        action_buffers = [ReplayBuffer(action_buffers_size) for _ in range(num_actions)]
        for t in tqdm(range(total_timesteps)):

            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True


            if thompson:
                # for each action sample one of the num_models samples of w
                model_idx = np.random.randint(0, num_models, size=num_actions)
                cur_w = np.zeros((num_actions, feat_dim))
                for i in range(num_actions):
                    cur_w[i] = w_sample[i, model_idx[i]]
                action, estimate = act(np.array(obs)[None], cur_w[None])
                actions_hist[int(action)] += 1
                actions_hist_total[int(action)] += 1
            else:
                action, estimate = act(np.array(obs)[None], update_eps=update_eps, **kwargs)
            env_action = action
            reset = False
            new_obs, unclipped_rew, done_list, _ = env.step(env_action)
            done, real_done = done_list
            rew = np.sign(unclipped_rew)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            action_buffers[action].add(obs, action, rew, new_obs, float(done))
            if action_buffers[action]._next_idx == 0:
                obses_a, actions_a, rewards_a, obses_tp1_a, dones_a = replay_buffer.get_samples([i for i in range(action_buffers_size)])
                phiphiT_a, phiY_a = blr_ops_old(obses_a, actions_a, rewards_a, obses_tp1_a, dones_a)
                phiphiT[action] += phiphiT_a
                phiY[action] += phiY_a
            obs = new_obs
            episode_rewards[-1] += rew
            # episode_Q_estimates[-1] += estimate
            unclipped_episode_rewards[-1] += unclipped_rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                # episode_Q_estimates.append(0.0)
                reset = True
            if real_done:
                unclipped_episode_rewards.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if thompson:
                if t > learning_starts and t % (blr_params.update_w*target_network_update_freq) == 0:
                    llw = sess.run(blr_additions['last_layer_weights'])
                    phiphiT, phiY, phiphiT0, last_layer_weights = BayesRegression(phiphiT,phiY,replay_buffer,
                                                                 blr_additions['feature_extractor'],
                                                                 blr_additions['target_feature_extractor'], num_actions,
                                                                 blr_params,w_mu, w_cov,
                                                                 llw,
                                                                 prior=prior, blr_ops=blr_additions['blr_ops'],
                                                                 sdp_ops=blr_additions['sdp_ops'],
                                                                 old_networks=blr_additions['old_networks'],
                                                                 blr_counter=blr_counter, old_feat=blr_additions['old_feature_extractor'])
                    if seed is not None:
                        print('seed is {}'.format(seed))
                    blr_additions['update_old']()
                    # if blr_additions['old_networks'] is not None:
                    #     if blr_counter == 0:
                    #         for key in blr_additions['old_networks'].keys():
                    #             blr_additions['old_networks'][key]["update"]()
                    #             blr_additions['old_networks'][key]["phiphiT"] = phiphiT
                    #     else:
                    #         blr_additions['old_networks'][blr_counter % 5]["update"]()
                    #         blr_additions['old_networks'][blr_counter % 5]["phiphiT"] = phiphiT
                    blr_counter += 1

            if thompson:
                if t > 0 and t % blr_params.sample_w == 0:
                    # sampling num_models samples of w
                    # print(actions_hist)
                    actions_hist = [0. for _ in range(num_actions)]
                    for i in range(num_actions):
                        if prior == 'no prior' or last_layer_weights is None:
                            cov = np.linalg.inv(phiphiT[i])
                            mu = np.array(np.dot(cov,phiY[i]))
                        elif prior == 'last layer':
                            cov = np.linalg.inv(phiphiT[i])
                            mu = np.array(np.dot(cov,(phiY[i] + (1/blr_params.sigma)*last_layer_weights[:,i])))
                        elif prior == 'single sdp':
                            try:
                                cov = np.linalg.inv(phiphiT[i] + phiphiT0)
                            except:
                                print("singular matrix using pseudo inverse")
                                cov = np.linalg.pinv(phiphiT[i] + phiphiT0)
                            mu = np.array(np.dot(cov,(phiY[i] + np.dot(phiphiT0, last_layer_weights[:,i]))))
                        elif prior == 'sdp' or prior == 'linear':
                            print(prior)
                            try:
                                cov = np.linalg.inv(phiphiT[i] + phiphiT0[i])
                            except:
                                print("singular matrix")
                                cov = np.linalg.pinv(phiphiT[i] + phiphiT0[i])
                            mu = np.array(np.dot(cov,(phiY[i] + np.dot(phiphiT0[i], last_layer_weights[:,i]))))
                        else:
                            print("No valid prior")
                            exit(0)


                        for j in range(num_models):
                            try:
                                w_sample[i, j] = np.random.multivariate_normal(mu, blr_params.sigma*cov)
                            except:
                                w_sample[i, j] = mu
                    # w_norms = [np.linalg.norm(w_sample[i]) for i in range(num_actions)]

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                # print(update_target)
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
            mean_100ep_reward_unclipped = round(np.mean(unclipped_episode_rewards[-101:-1]), 1)
            mean_10ep_reward_unclipped = round(np.mean(unclipped_episode_rewards[-11:-1]), 1)
            # mean_100ep_est = round(np.mean(episode_Q_estimates[-101:-1]), 1)
            # mean_10ep_est = round(np.mean(episode_Q_estimates[-11:-1]), 1)
            num_episodes = len(episode_rewards)
            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            if t % 10000 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                logger.record_tabular("mean 100 unclipped episode reward", mean_100ep_reward_unclipped)
                logger.record_tabular("mean 10 unclipped episode reward", mean_10ep_reward_unclipped)
                # logger.record_tabular("mean 100 episode Q estimates", mean_100ep_est)
                # logger.record_tabular("mean 10 episode Q estimates", mean_10ep_est)
                logger.dump_tabular()
                if t % 7 == 0:
                    print("len(unclipped_episode_rewards)")
                    print(len(unclipped_episode_rewards))
                    print("len(episode_rewards)")
                    print(len(episode_rewards))

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act
