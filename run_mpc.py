import deeprl_hw3.arm_env as arm_env
import deeprl_hw3.controllers as controllers
import deeprl_hw3.fast_ilqr as ilqr
import gym
import numpy as np
import time


def run_ilqr_controller(env, render_flag, sim_env, tN):
    env.reset()
    if render_flag:
        env.render()
        # time.sleep(0.01)

    total_reward = 0
    num_steps = 0

    q = [env.q]
    dq = [env.dq]
    u = []
    total_cost = []

    while True:
        U, cost = ilqr.calc_ilqr_input(env, sim_env, tN)
        total_cost.append(cost)

        action = U[0]
        u.append(action)
        _, reward, is_terminal, _ = env._step(action)
        q.append(env.q)
        dq.append(env.dq)
        if render_flag:
            env.render()
            # time.sleep(0.01)

        total_reward += reward
        num_steps += 1
        if is_terminal:
            break
    return q, dq, u, total_reward, num_steps


def main():
    # Create environment
    env = gym.make('TwoLinkArm-v0')
    sim_env = gym.make('TwoLinkArm-v0')
    # env = gym.make('TwoLinkArm-limited-torque-random-goal-v1')
    # sim_env = gym.make('TwoLinkArm-limited-torque-random-goal-v1')
    # env = gym.make('TwoLinkArm-limited-torque-v0')

    render_flag = True
    tN = 100

    q, dq, u, total_reward, num_steps = run_ilqr_controller(env, render_flag, sim_env, tN)

    print total_reward, num_steps


if __name__ == '__main__':
    main()
