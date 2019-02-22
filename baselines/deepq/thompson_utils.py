
import cvxpy as cvx
from datetime import datetime
from random import shuffle
import numpy as np
from tqdm import tqdm
import tensorflow as tf

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
    outer_k_matrix = [None for _ in range(num_actions)]
    pseudo_count_k_matrix = [None for _ in range(num_actions)]
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
            if outer_k_matrix[k] is None:
                outer_k_matrix[k] = outer_k.transpose([0,2,1]).reshape(nk,feat_dim*feat_dim)
                pseudo_count_k_matrix[k] = pseudo_count_k
            else:
                outer_k_matrix[k] = np.concatenate([outer_k_matrix[k], outer_k.transpose([0,2,1]).reshape(nk,feat_dim*feat_dim)], axis=0)
                pseudo_count_k_matrix[k] = np.concatenate([pseudo_count_k_matrix[k], pseudo_count_k], axis=0)
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
                res = np.linalg.lstsq(outer_k_matrix[a], pseudo_count_k_matrix[a])
                Xnp = res[0].reshape((feat_dim,feat_dim))
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

def information_transfer_linear(phiphiT, dqn_feat, old_feat, num_actions, feat_dim,actions_buffers, phiY, sdp_ops):
    # n_samples = min([300000, len(replay_buffer)])
    # idxes = [i for i in range(n_samples)]
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes)
    # obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = [], [], [], [], []
    # idxes_per_a = []

    # for new liear approach
    phiphiT_inv = np.zeros_like(phiphiT)
    print("phiphiT inv")
    for i in range(num_actions):
        phiphiT_inv[i] = np.linalg.pinv(phiphiT[i])

    obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = actions_buffers
    d1 = datetime.now()
    # for i in range(num_actions):
    #     obses_t_per_a.append(obses_t[actions == i])
    #     actions_per_a.append(actions[actions == i])
    #     rewards_per_a.append(rewards[actions == i])
    #     obses_tp1_per_a.append(obses_tp1[actions == i])
    #     dones_per_a.append(dones[actions == i])
    #     ix = [j for j in range(obses_t[actions == i].shape[0])]
    #     shuffle(ix)
    #     idxes_per_a.append(ix)

    phiphiT0 = np.zeros_like(phiphiT)
    mu0 = np.zeros_like(phiY)

    phiphiT0_inv_alternative = np.zeros_like(phiphiT)
    outer_k_matrix = [None for _ in range(num_actions)]
    pseudo_count_k_matrix = [None for _ in range(num_actions)]
    for i in range(num_actions):
        phi_m = None
        xi_m = None
        mi = obses_t_per_a[i].shape[0]
        M = min([1000, mi])
        print("linear prior for action {}, samples: {}".format(i,M))
        if M < feat_dim:
            phiphiT0[i] = 1/0.001 * np.eye(feat_dim)
            continue
        mini_batch_size = 1000
        for m in range(M // mini_batch_size + 1):
            start_idx = m*mini_batch_size
            end_idx = min([(m+1)*mini_batch_size, M])
            if start_idx == end_idx:
                continue
            phi_t = old_feat(obses_t_per_a[i][start_idx:end_idx][None]).T
            xi_t = dqn_feat(obses_t_per_a[i][start_idx:end_idx][None]).T

            # additions for new linear method
            ni = xi_t.shape[0]
            pseudo_count_k, outer_k = sdp_ops(obses_t_per_a[i][start_idx:end_idx][None], obses_t_per_a[i][start_idx:end_idx][None], np.tile(phiphiT_inv[i],(ni,1,1)))
            if outer_k_matrix[i] is None:
                outer_k_matrix[i] = outer_k.transpose([0,2,1]).reshape(ni,feat_dim*feat_dim)
                pseudo_count_k_matrix[i] = pseudo_count_k
            else:
                outer_k_matrix[i] = np.concatenate([outer_k_matrix[i], outer_k.transpose([0,2,1]).reshape(ni,feat_dim*feat_dim)], axis=0)
                pseudo_count_k_matrix[i] = np.concatenate([pseudo_count_k_matrix[i], pseudo_count_k], axis=0)

            if phi_m is None:
                phi_m = phi_t
            else:
                phi_m = np.concatenate([phi_m, phi_t],axis=-1)
            if xi_m is None:
                xi_m = xi_t
            else:
                xi_m = np.concatenate([xi_m, xi_t],axis=-1)
        phi_m_inv = np.linalg.pinv(phi_m)
        # xi_m_phi_m_inv = tf.matmul(xi_m, phi_m_inv).eval()
        # phiphiT0[i] = tf.matmul(tf.matmul(xi_m_phi_m_inv, phiphiT[i]).eval(), xi_m_phi_m_inv.T).eval()# + 1/0.001 * np.eye(feat_dim)
        phiphiT0[i] = xi_m @ phi_m_inv @ phiphiT[i] @ phi_m_inv.T @ xi_m.T# + 1/0.001 * np.eye(feat_dim)

        # TODO: currently using last layer weights, if you want to use linear prior for expectation remove comment
        # phiphiT_inv = np.linalg.pinv(phiphiT[i])
        # xi_m_inv = np.linalg.pinv(xi_m)
        # phi_m_xi_m_inv = tf.matmul(phi_m, xi_m_inv).eval()
        # mu0[i] = tf.matmul(tf.matmul(phiphiT_inv, phiY[i][..., None]).eval().T, phi_m_xi_m_inv).eval()
    d2 = datetime.now()
    print("total time for linear prior")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    return phiphiT0, mu0.T

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

def BayesRegressionOld(phiphiT, phiY, replay_buffer, dqn_feat, target_dqn_feat, num_actions, blr_param,
                    w_mu, w_cov, last_layer_weights, prior="sdp", blr_ops=None, sdp_ops=None, old_networks=None, blr_counter=None,old_feat=None):
    # dqn_ and target_dqn_ are feature extractors for the dqn and target dqn respectivley
    # in the neural linear settings target are the old features and dqn are the new features
    # last_layer_weights are the target last layer weights
    feat_dim = blr_param.feat_dim

    n_samples = min([200000, len(replay_buffer)])
    idxes = [i for i in range(n_samples)]
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes)
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
        phiphiT0, last_layer_weights = information_transfer_linear(phiphiT, dqn_feat, old_feat,num_actions,feat_dim, replay_buffer, phiY)
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

    YY = np.zeros(num_actions)
    a = np.ones(num_actions)*blr_param.a0
    b = np.ones(num_actions)*blr_param.b0
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes[:n_samples])
    n = np.zeros(num_actions)
    action_rewards = [0. for _ in range(num_actions)]

    mini_batch_size = 100*num_actions
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
            phiphiTk, phiYk, YYk = blr_ops(obs_t[action == k], action[action == k], reward[action == k], obs_tp1[action == k], done[action == k])
            phiphiT[k] += phiphiTk
            phiY[k] += phiYk
            YY[k] += YYk
            n[k] += obs_t[action == k].shape[0]
            action_rewards[k] += sum(reward[action == k])
        for k in range(num_actions):
            precision = phiphiT[k] + phiphiT0[k]
            cov = np.linalg.pinv(precision)
            mu = np.array(np.dot(cov,(phiY[k] + np.dot(phiphiT0[k], last_layer_weights[:,k]))))
            a[k] += 0.5*n[k]
            b_upd = 0.5 * YY[k]
            b_upd += 0.5 * np.dot(last_layer_weights[:,k].T, np.dot(phiphiT0[k], last_layer_weights[:,k]))
            b_upd -= 0.5 * np.dot(mu.T, np.dot(precision, mu))
            b[k] += b_upd
    print(n, np.sum(n))
    return phiphiT, phiY, phiphiT0, last_layer_weights, YY, a, b

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

def BayesRegression(phiphiT, phiY, replay_buffer, dqn_feat, target_dqn_feat, num_actions, blr_param,
                    w_mu, w_cov, last_layer_weights, prior="sdp", blr_ops=None, sdp_ops=None, old_networks=None,
                    blr_counter=None, old_feat=None,a=None):

    # dqn_ and target_dqn_ are feature extractors for the dqn and target dqn respectivley
    # in the neural linear settings target are the old features and dqn are the new features
    # last_layer_weights are the target last layer weights
    feat_dim = blr_param.feat_dim

    n_samples = min([blr_param.batch_size, len(replay_buffer)])
    # idxes = [i for i in range(n_samples)]
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes)
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.n_samples_per_action(n=20000)
    # obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = [], [], [], [], []

    obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a = replay_buffer.n_samples_per_action(n=20000)
    # idxes_per_a = []
    # for i in range(num_actions):
    #     obses_t_per_a.append(obses_t[actions == i])
    #     actions_per_a.append(actions[actions == i])
    #     rewards_per_a.append(rewards[actions == i])
    #     obses_tp1_per_a.append(obses_tp1[actions == i])
    #     dones_per_a.append(dones[actions == i])
    #     ix = [j for j in range(obses_t[actions == i].shape[0])]
    #     shuffle(ix)
    #     idxes_per_a.append(ix)

    actions_buffers = [obses_t_per_a, actions_per_a, rewards_per_a, obses_tp1_per_a, dones_per_a]#, idxes_per_a]

    phiphiT0 = None
    if prior == "no prior":
        print("no prior")
        phiY *= (1 - blr_param.alpha) * 0
        phiphiT *= (1 - blr_param.alpha) * 0
        a = np.ones(num_actions) * blr_param.a0
        phiphiT0 = np.zeros_like(phiphiT)
        for i in range(num_actions):
            phiphiT0[i] = (1 / blr_param.sigma) * np.eye(feat_dim)
        last_layer_weights = np.zeros_like(last_layer_weights)
    elif prior == "linear":
        print("linear")
        phiphiT0, _ = information_transfer_linear(phiphiT, dqn_feat, old_feat, num_actions, feat_dim,
                                                                   actions_buffers, phiY, sdp_ops)
        phiphiT *= (1 - blr_param.alpha) * 0
        phiY *= (1 - blr_param.alpha) * 0
    elif prior == "decay":
        print("simple prior")
        phiY *= (1 - blr_param.alpha)
        phiphiT *= (1 - blr_param.alpha)
    elif prior == "last layer":
        print("last layer weights only prior")
        phiY *= (1 - blr_param.alpha) * 0
        phiphiT *= (1 - blr_param.alpha) * 0
        for i in range(num_actions):
            phiphiT[i] = (1 / blr_param.sigma) * np.eye(feat_dim)
    elif prior == "sdp":
        print("SDP prior")
        phiphiT0, cov0 = information_transfer_new(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, 300 * num_actions,
                                                  num_actions, feat_dim, sdp_ops)
        for j in range(num_actions):
            print("old phiphiT[{}] new features norm:".format(j))
            print(np.linalg.norm(phiphiT0[j]))
            print("old cov0[{}] new features norm:".format(j))
            print(np.linalg.norm(cov0[j]))
        phiphiT *= (1 - blr_param.alpha) * 0
        phiY *= (1 - blr_param.alpha) * 0
    elif prior == "single sdp":
        print("single SDP prior")
        # phiphiT0 = 1/blr_param.sigma * np.eye(feat_dim)
        # phiphiT0 = None
        # if np.any(phiphiT != np.zeros_like(phiphiT)):
        print("using 600")
        phiphiT0, cov0 = information_transfer_single(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, 600, num_actions,
                                                     feat_dim, sdp_ops, old_networks, blr_counter)
        phiphiT *= (1 - blr_param.alpha) * 0
        phiY *= (1 - blr_param.alpha) * 0

    YY = np.zeros(num_actions)
    # a = np.ones(num_actions) * blr_param.a0
    b = np.ones(num_actions) * blr_param.b0
    # obses_t, actions, rewards, obses_tp1, dones = replay_buffer.get_samples(idxes[:n_samples])
    n = np.zeros(num_actions)
    action_rewards = [0. for _ in range(num_actions)]

    for i in range(num_actions):
        mi = obses_t_per_a[i].shape[0]
        M = min([20000, mi])
        print("BLR n_samples for action {}: {}".format(i,M))
        if M < feat_dim:
            print("very low samples for action {}".format(i))
        mini_batch_size = 2048
        for m in range(M // mini_batch_size + 1):
            start_idx = m*mini_batch_size
            end_idx = min([(m+1)*mini_batch_size, M])
            # phiphiTm, phiYm, YYm = blr_ops(obses_t_per_a[i][idxes_per_a[i][start_idx:end_idx]],
            #                                actions_per_a[i][idxes_per_a[i][start_idx:end_idx]],
            #                                rewards_per_a[i][idxes_per_a[i][start_idx:end_idx]],
            #                                obses_tp1_per_a[i][idxes_per_a[i][start_idx:end_idx]],
            #                                dones_per_a[i][idxes_per_a[i][start_idx:end_idx]])
            phiphiTm, phiYm, YYm = blr_ops(obses_t_per_a[i][start_idx:end_idx],
                                           actions_per_a[i][start_idx:end_idx],
                                           rewards_per_a[i][start_idx:end_idx],
                                           obses_tp1_per_a[i][start_idx:end_idx],
                                           dones_per_a[i][start_idx:end_idx])
            phiphiT[i] += phiphiTm
            phiY[i] += phiYm
            YY[i] += YYm
            n[i] += end_idx - start_idx

            action_rewards[i] += sum(rewards_per_a[i][start_idx:end_idx])

    for k in range(num_actions):
        precision = phiphiT[k] + phiphiT0[k]
        cov = np.linalg.pinv(precision)
        mu = np.array(np.dot(cov, (phiY[k] + np.dot(phiphiT0[k], last_layer_weights[:, k]))))
        a[k] += 0.5 * n[k]
        b_upd = 0.5 * YY[k]
        b_upd += 0.5 * np.dot(last_layer_weights[:, k].T, np.dot(phiphiT0[k], last_layer_weights[:, k]))
        b_upd -= 0.5 * np.dot(mu.T, np.dot(precision, mu))
        b[k] += b_upd

    if prior == "no prior":
        phiphiT += phiphiT0

    print(n, np.sum(n))
    print(action_rewards)
    return phiphiT, phiY, phiphiT0, last_layer_weights, YY, a, b

