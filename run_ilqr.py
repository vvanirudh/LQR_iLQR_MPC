import deeprl_hw3.arm_env as arm_env
import deeprl_hw3.controllers as controllers
import deeprl_hw3.ilqr as ilqr
import gym
import numpy as np
import time
import matplotlib.pyplot as plt


def plotthem(env_name, q, dq, u, num_steps, total_reward, total_cost):

    # create a running index for plot
    x = np.arange(1, num_steps+2, 1)
    q1 = np.array(q)
    dq1 = np.array(dq)
    u1 = np.array(u)

    plt.figure(figsize=(15, 15))
    p1 = plt.subplot(411)
    p1.plot(x, q1[:, 0])
    p1.plot(x, q1[:, 1])
    p1.legend([' = q[0]', ' = q[1]'], loc='upper right')
    p1.set_title('Fast ILQR Environment : '+ENV_NAME+'\n Total reward = '+str(total_reward)+' Number of steps= '+str(num_steps)+'\n\n q, vs number of steps')

    p2 = plt.subplot(412)
    p2.plot(x, dq1[:, 0])
    p2.plot(x, dq1[:, 1])
    p2.legend([' = dq[0]', ' = dq[1]'], loc='upper right')
    p2.set_title('dq vs number of steps ')

    p3 = plt.subplot(413)
    p3.plot(x[:-1], u1[:, 0])
    p3.plot(x[:-1], u1[:, 1])
    p3.legend([' = u[0]', ' = u[1]'], loc='upper right')
    p3.set_title('u values values vs number of steps')

    p4 = plt.subplot(414)
    p4.plot(range(len(total_cost)), total_cost)
    p4.legend(['Cost of trajectory'], loc='upper right')
    p4.set_title('Total cost vs ILQR iterations')

    # plt.show()
    plt.savefig(env_name + '_fast_ilqr.png')
    plt.gcf().clear()
    plt.close()
    return plt


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
    terminal_flag = False

    while True:
        U, cost_list = ilqr.calc_ilqr_input(env, sim_env, tN)
        total_cost = total_cost + cost_list

        for i in range(tN):
            action = U[i]
            u.append(np.copy(action))
            _, reward, is_terminal, _ = env._step(action)
            q.append(np.copy(env.q))
            dq.append(np.copy(env.dq))
            if render_flag:
                env.render()
                time.sleep(0.01)

            total_reward += reward
            num_steps += 1
            if is_terminal:
                terminal_flag = True
                break
        if terminal_flag:
            break
    return q, dq, u, total_reward, num_steps, total_cost


# ENV_NAME = 'TwoLinkArm-v0'
# ENV_NAME = 'TwoLinkArm-limited-torque-v0'
# ENV_NAME = 'TwoLinkArm-random-goal-v0'
# ENV_NAME = 'TwoLinkArm-limited-torque-random-goal-v0'
# ENV_NAME = 'TwoLinkArm-v1'
# ENV_NAME = 'TwoLinkArm-limited-torque-v1'
# ENV_NAME = 'TwoLinkArm-random-goal-v1'
# ENV_NAME = 'TwoLinkArm-limited-torque-random-goal-v1'
ENVS = ['TwoLinkArm-v0', 'TwoLinkArm-v1']

for ENV_NAME in ENVS:
    # Create environment
    env = gym.make(ENV_NAME)
    sim_env = gym.make(ENV_NAME)

    render_flag = True
    tN = 100

    q, dq, u, total_reward, num_steps, total_cost = run_ilqr_controller(env, render_flag, sim_env, tN)

    plotthem(ENV_NAME, q, dq, u, num_steps, total_reward, total_cost)
