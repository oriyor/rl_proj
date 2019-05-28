import datetime

import gym
import torch.optim as optim

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


def test_frame_history(num_timesteps, frame_lst=[1, 4, 10, 100]):
    for frame in frame_lst:
        # Get Atari games.
        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0  # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)
        env.reset()

        runname = "test_frame_" + str(frame) + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        optimizer_spec = OptimizerSpec(constructor=optim.RMSprop, kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS))

        exploration_schedule = LinearSchedule(1000000, 0.1)

        dqn_learning(
            env=env,
            q_func=DQN,
            runname=runname,
            optimizer_spec=optimizer_spec,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion2(num_timesteps),
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=frame,
            target_update_freq=TARGET_UPDATE_FREQ
        )


def test_scheduler_history(num_timesteps):
    schedulers = {"linear": LinearSchedule(1000000, 0.1),
                  "const": ConstantSchedule(0.05),
                  "non": ConstantSchedule(0.0),
                  "piecewise": PiecewiseSchedule(
                      [(0.5e6, 0.1), (1e6, 0.075), (1.5e6, 0.05), (2e6, 0.025), (3e6, 0.001)],
                      outside_value=0)}
    for name, exploration_schedule in schedulers.items():
        # Get Atari games.
        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0  # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)
        env.reset()

        runname = "test_scheduler_" + str(name) + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        optimizer_spec = OptimizerSpec(constructor=optim.RMSprop, kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS))

        dqn_learning(
            env=env,
            q_func=DQN,
            runname=runname,
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


def test_optimizer(num_timesteps):
    optimizers = [("RMSProp", optim.RMSprop, dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)),
                  ("Adam_1e-4", optim.Adam, dict(lr=1e-4)),
                  ("Adam_1e-3", optim.Adam, dict(lr=1e-3)),
                  ("SGD", optim.SGD, dict(lr=1e-3))]
    for optimizer_name, optimizer, optim_kwargs in optimizers.items():
        # Get Atari games.
        benchmark = gym.benchmark_spec('Atari40M')

        # Change the index to select a different game.
        task = benchmark.tasks[3]

        # Run training
        seed = 0  # Use a seed of zero (you may want to randomize the seed!)
        env = get_env(task, seed)
        env.reset()

        runname = "test_scheduler_" + str(optimizer_name) + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        optimizer_spec = OptimizerSpec(constructor=optimizer, kwargs=optim_kwargs)

        exploration_schedule = LinearSchedule(1000000, 0.1)

        dqn_learning(
            env=env,
            q_func=DQN,
            runname=runname,
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


def main(env, num_timesteps):
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learning(
        env=env,
        q_func=DQN,
        runname="Normal_run",
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


if __name__ == '__main__':
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)

    main(env, 2e6)

    #test_scheduler_history(2e6)
