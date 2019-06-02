import sys

import gym
import torch.optim as optim
import numpy as np

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learning
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule, ConstantSchedule, PiecewiseSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01


def stopping_criterion(num_timesteps):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return lambda env: get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps


def stopping_criterion2(num_timesteps):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return lambda t: t >= num_timesteps


def q1_run(num_timesteps):
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed, expt_dir='tmp/gym-results2')

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learning(
        env=env,
        q_func=DQN,
        runname="normal_run",
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion2(num_timesteps),
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ
    )


def q2_run(num_timesteps):
    schedulers = {"no_explore": ConstantSchedule(0.1),
                  "delayed_decay": PiecewiseSchedule([(0, 1.0), (0.25e6, 1.0), (1.25e6, 0.1)], outside_value=0.1),
                  "slower_decay": LinearSchedule(1500000, 0.1)}

    for name, exploration_schedule in schedulers.items():
        # Get Atari games.
        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0  # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)
        env.reset()

        optimizer_spec = OptimizerSpec(constructor=optim.RMSprop, kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS))

        dqn_learning(
            env=env,
            q_func=DQN,
            runname=name,
            optimizer_spec=optimizer_spec,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion2(num_timesteps),
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ
        )


def bonus_run(num_timesteps):
    def make_range_black(arr: np.ndarray, start, end):
        arr[:, start:end, :] = 0

    frame_filters = {"no_left_side": lambda x: make_range_black(x, 0, x.shape[1] // 4),
                     "no_middle_side": lambda x: make_range_black(x, x.shape[1] // 4, x.shape[1] // 2), }

    for name, frame_filter in frame_filters.items():
        # Get Atari games.
        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0  # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)
        env.reset()

        optimizer_spec = OptimizerSpec(constructor=optim.RMSprop, kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS))

        dqn_learning(
            env=env,
            q_func=DQN,
            runname=name,
            frame_filter=frame_filter,
            optimizer_spec=optimizer_spec,
            exploration=LinearSchedule(1000000, 0.1),
            stopping_criterion=stopping_criterion2(num_timesteps),
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ
        )


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Should provide an argument for which part to run")
    arg_run = sys.argv[1]

    if arg_run == "q1":
        q1_run(4e6)
    elif arg_run == "q2":
        q2_run(3e6)
    elif arg_run == "bonus":
        bonus_run(3e6)
