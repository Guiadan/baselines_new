from baselines.deepq.deepq import debug_flag as debug
def atari():
    if debug:
        return dict(
            network='conv_only',
            lr=1e-4,
            buffer_size=100000,#10000
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=5000,#10000
            target_network_update_freq=10000,
            gamma=0.99,
            prioritized_replay=False,
            prioritized_replay_alpha=0.6,
            checkpoint_freq=10000,
            checkpoint_path=None,
            dueling=False
        )
    else:
        return dict(
            network='conv_only',
            lr=1e-4,#1e-4
            buffer_size=1000000,#1000000
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=50000,#10000
            target_network_update_freq=5000,
            gamma=0.99,
            prioritized_replay=False,
            prioritized_replay_alpha=0.6,
            checkpoint_freq=10000,
            checkpoint_path=None,
            dueling=False
        )

def retro():
    return atari()

