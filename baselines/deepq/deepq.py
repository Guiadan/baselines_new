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
from baselines.deepq.thompson_utils import BayesRegression

#additions
from scipy.stats import invgamma
from tqdm import tqdm
debug_flag = True
structred_learning = False
first_time = True

class BLRParams(object):
    def __init__(self):
        self.sigma = 10 #0.001 W prior variance
        self.sigma_n = 1 # noise variance
        self.alpha = .01 # forgetting factor
        if debug_flag:
            self.update_w = 1 # multiplied by update target frequency
            self.sample_w = 1000
        else:
            self.sample_w = 10000
            self.update_w = 5 # multiplied by update target frequency
        self.batch_size = 1000000# batch size to do blr from
        self.gamma = 0.99 #dqn gamma
        self.feat_dim = 128 #256
        self.first_time = True
        self.no_prior = True
        self.a0 = 7
        self.b0 = 60



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


    #deep mind optimizer
    # dm_opt = tf.train.RMSPropOptimizer(learning_rate=0.00025,decay=0.95,momentum=0.0,epsilon=0.00001,centered=True)
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

        phiphiT = np.zeros((num_actions,feat_dim,feat_dim),dtype=np.float32)
        phiphiT_inv = np.zeros((num_actions,feat_dim,feat_dim), dtype=np.float32)
        for i in range(num_actions):
            phiphiT[i] = (1/blr_params.sigma)*np.eye(feat_dim)
            phiphiT_inv[i] = blr_params.sigma*np.eye(feat_dim)
        old_phiphiT_inv = [phiphiT_inv for i in range(5)]

        phiY = np.zeros((num_actions, feat_dim), dtype=np.float32)
        YY = np.zeros(num_actions)

        model_idx = np.random.randint(0,num_models,size=num_actions)
        w_norms = [np.linalg.norm(w_sample[i,model_idx[i]]) for i in range(num_actions)]
        blr_ops = blr_additions['blr_ops']
        blr_ops_old = blr_additions['blr_ops_old']

        last_layer_weights = np.zeros((feat_dim, num_actions))
        phiphiT0 = np.copy(phiphiT)

        invgamma_a = [blr_params.a0 for _ in range(num_actions)]
        invgamma_b = [blr_params.a0 for _ in range(num_actions)]
    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()
    if thompson:
        blr_additions['update_old']()
        blr_additions['update_old_target']()
        if blr_additions['old_networks'] is not None:
            for key in blr_additions['old_networks'].keys():
                blr_additions['old_networks'][key]["update"]()

    episode_rewards = [0.0]
    # episode_Q_estimates = [0.0]
    unclipped_episode_rewards = [0.0]
    eval_rewards = [0.0]


    old_networks_num = 5
    # episode_pseudo_count = [[0.0] for i in range(old_networks_num)]
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
        action_buffers_size = 1024
        action_buffers = [ReplayBuffer(action_buffers_size) for _ in range(num_actions)]
        eval_flag = False
        eval_counter = 0
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
                phiphiT_a, phiY_a, YY_a = blr_ops_old(obses_a, actions_a, rewards_a, obses_tp1_a, dones_a)
                phiphiT[action] += phiphiT_a
                phiY[action] += phiY_a
                YY[action] += YY_a

                precision = phiphiT[action] + phiphiT0[action]
                cov = np.linalg.pinv(precision)
                mu = np.array(np.dot(cov,(phiY[action] + np.dot(phiphiT0[action], last_layer_weights[:,action]))))
                invgamma_a[action] += 0.5*action_buffers_size
                b_upd = 0.5 * YY[action]
                b_upd += 0.5 * np.dot(last_layer_weights[:,action].T, np.dot(phiphiT0[action], last_layer_weights[:,action]))
                b_upd -= 0.5 * np.dot(mu.T, np.dot(precision, mu))
                invgamma_b[action] += b_upd

                # old_phiphiT_inv_a = [np.tile(oppTi[action], (action_buffers_size,1,1)) for oppTi in old_phiphiT_inv]
                # old_pseudo_count = blr_additions['old_pseudo_counts'](obses_a, *old_phiphiT_inv_a)
                # old_pseudo_count = np.sum(old_pseudo_count, axis=-1)
                # for i in range(old_networks_num):
                #     idx = ((blr_counter-1)-i) % old_networks_num # arrange networks from newest to oldest
                #     episode_pseudo_count[i][-1] += old_pseudo_count[idx]

            # if real_done:
            #     for a in range(num_actions):
            #         if action_buffers[a]._next_idx != 0:
            #             obses_a, actions_a, rewards_a, obses_tp1_a, dones_a = replay_buffer.get_samples([i for i in range(action_buffers[a]._next_idx)])
            #             nk = obses_a.shape[0]
            #
            #             # old_phiphiT_inv_a = [np.tile(oppTi[action],(nk,1,1)) for oppTi in old_phiphiT_inv]
            #             # old_pseudo_count = blr_additions['old_pseudo_counts'](obses_a, *old_phiphiT_inv_a)
            #             # old_pseudo_count = np.sum(old_pseudo_count, axis=-1)
            #             # for i in range(old_networks_num):
            #             #     idx = ((blr_counter-1)-i) % old_networks_num # arrange networks from newest to oldest
            #             #     episode_pseudo_count[i][-1] += old_pseudo_count[idx]
            #
            #             phiphiT_a, phiY_a, YY_a = blr_ops_old(obses_a, actions_a, rewards_a, obses_tp1_a, dones_a)
            #             phiphiT[a] += phiphiT_a
            #             phiY[a] += phiY_a
            #             YY[a] += YY_a
            #
            #             action_buffers[a]._next_idx = 0


            obs = new_obs
            episode_rewards[-1] += rew
            # episode_Q_estimates[-1] += estimate
            unclipped_episode_rewards[-1] += unclipped_rew

            # if t % 250000 == 0 and t > 0:
            #     eval_flag = True

            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                # episode_Q_estimates.append(0.0)
                reset = True
                if real_done:
                    unclipped_episode_rewards.append(0.0)
                    # for i in range(old_networks_num):
                    #     episode_pseudo_count[i].append(0.0)
                    # every time full episode ends run eval episode
                    # if eval_flag:
                    #     real_done = False
                    #     eval_rewards = [0.0]
                    #     for te in range(125000):
                    #         action, _ = blr_additions['eval_act'](np.array(obs)[None])
                    #         new_obs, unclipped_rew, done_list, _ = env.step(action)
                    #         done, real_done = done_list
                    #         eval_rewards[-1] += unclipped_rew
                    #         obs = new_obs
                    #         if done:
                    #             obs = env.reset()
                    #         if real_done:
                    #             eval_rewards.append(0.0)
                    #     obs = env.reset()
                    #     eval_rewards.pop()
                    #     mean_reward_eval = round(np.mean(eval_rewards), 1)
                    #     logger.record_tabular("mean eval episode reward", mean_reward_eval)
                    #     logger.dump_tabular()
                    #     eval_flag = False
                    eval_counter += 1
                    if eval_counter % 10 == 0:
                        if t > learning_starts:
                            real_done = False
                            while not real_done:
                                action, _ = blr_additions['eval_act'](np.array(obs)[None])
                                new_obs, unclipped_rew, done_list, _ = env.step(action)
                                done, real_done = done_list
                                eval_rewards[-1] += unclipped_rew
                                obs = new_obs
                            eval_rewards.append(0.0)
                            obs = env.reset()

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
                    phiphiT_inv = np.zeros_like(phiphiT)
                    for i in range(num_actions):
                        try:
                            phiphiT_inv[i] = np.linalg.inv(phiphiT[i])
                        except:
                            phiphiT_inv[i] = np.linalg.pinv(phiphiT[i])
                    old_phiphiT_inv[blr_counter % 5] = phiphiT_inv
                    llw = sess.run(blr_additions['last_layer_weights'])
                    phiphiT, phiY, phiphiT0, last_layer_weights, YY, invgamma_a, invgamma_b = BayesRegression(phiphiT,phiY,replay_buffer,
                                                                 blr_additions['feature_extractor'],
                                                                 blr_additions['target_feature_extractor'], num_actions,
                                                                 blr_params,w_mu, w_cov,
                                                                 llw,
                                                                 prior=prior, blr_ops=blr_additions['blr_ops'],
                                                                 sdp_ops=blr_additions['sdp_ops'],
                                                                 old_networks=blr_additions['old_networks'],
                                                                 blr_counter=blr_counter, old_feat=blr_additions['old_feature_extractor'])
                    blr_counter += 1
                    if seed is not None:
                        print('seed is {}'.format(seed))
                    blr_additions['update_old']()
                    blr_additions['update_old_target']()
                    if blr_additions['old_networks'] is not None:
                        blr_additions['old_networks'][blr_counter % 5]["update"]()

            if thompson:
                if t > 0 and t % blr_params.sample_w == 0:
                    # sampling num_models samples of w
                    print(actions_hist)
                    actions_hist = [0. for _ in range(num_actions)]
                    if t > 1000000:
                        adaptive_sigma = True
                    else:
                        adaptive_sigma = False
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
                            if adaptive_sigma:
                                sigma = invgamma_b[i] * invgamma.rvs(invgamma_a[i])
                            else:
                                sigma = blr_params.sigma
                            try:
                                w_sample[i, j] = np.random.multivariate_normal(mu, sigma*cov)
                            except:
                                w_sample[i, j] = mu
                    # w_norms = [np.linalg.norm(w_sample[i]) for i in range(num_actions)]
                        if t % 7 == 0:
                            print("action {}".format(i))
                            print("cov norm:")
                            print(np.linalg.norm(cov))
                            print("sigma")
                            print(sigma)
                            print("cov norm times sigma:")
                            print(np.linalg.norm(sigma*cov))

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                # print(update_target)
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
            mean_100ep_reward_unclipped = round(np.mean(unclipped_episode_rewards[-101:-1]), 1)
            mean_10ep_reward_unclipped = round(np.mean(unclipped_episode_rewards[-11:-1]), 1)
            mean_100ep_reward_eval = round(np.mean(eval_rewards[-101:-1]), 1)
            mean_10ep_reward_eval = round(np.mean(eval_rewards[-11:-1]), 1)
            # mean_100ep_est = round(np.mean(episode_Q_estimates[-101:-1]), 1)
            # mean_10ep_est = round(np.mean(episode_Q_estimates[-11:-1]), 1)
            num_episodes = len(episode_rewards)
            # mean_10ep_pseudo_count = [0.0 for _ in range(old_networks_num)]
            # mean_100ep_pseudo_count = [0.0 for _ in range(old_networks_num)]
            # for i in range(old_networks_num):
            #     mean_10ep_pseudo_count[i] = round(np.log(np.mean(episode_pseudo_count[i][-11:-1])), 1)
            #     mean_100ep_pseudo_count[i] = round(np.log(np.mean(episode_pseudo_count[i][-101:-1])), 1)


            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            if t % 10000 == 0 and t > 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                logger.record_tabular("mean 100 unclipped episode reward", mean_100ep_reward_unclipped)
                logger.record_tabular("mean 10 unclipped episode reward", mean_10ep_reward_unclipped)
                logger.record_tabular("mean 100 eval episode reward", mean_100ep_reward_eval)
                logger.record_tabular("mean 10 eval episode reward", mean_10ep_reward_eval)
                # for i in range(old_networks_num):
                #     logger.record_tabular("mean 10 episode pseudo count for -{} net".format(i+1), mean_10ep_pseudo_count[i])
                #     logger.record_tabular("mean 100 episode pseudo count for -{} net".format(i+1), mean_100ep_pseudo_count[i])
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
