import deeprl_hw3.controllers as controllers
import gym
import numpy as np
import time
import matplotlib.pyplot as plt


# plot q dq and u
def plotthem(q, dq, u, num_steps, total_reward):

    # create a running index for plot
    x = np.arange(1, num_steps+2, 1)
    q1 = np.array(q)
    dq1 = np.array(dq)
    u1 = np.array(u)

    p1 = plt.subplot(311)
    p1.plot(x, q1[:, 0])
    p1.plot(x, q1[:, 1])
    p1.legend([' = q[0]', ' = q[1]'], loc='upper right')
    p1.set_title('Environment : '+ENV_NAME+'\n Total reward = '+str(total_reward)+' Number of steps= '+str(num_steps)+'\n\n q, vs number of steps')

    p2 = plt.subplot(312)
    p2.plot(x, dq1[:, 0])
    p2.plot(x, dq1[:, 1])
    p2.legend([' = dq[0]', ' = dq[1]'], loc='upper right')
    p2.set_title('dq vs number of steps ')

    p3 = plt.subplot(313)
    p3.plot(x[:-1], u1[:, 0])
    p3.plot(x[:-1], u1[:, 1])
    p3.legend([' = u[0]', ' = u[1]'], loc='upper right')
    p3.set_title('u values values vs number of steps')
    plt.show()
    return plt


def run_lqr_controller(env, render_flag, sim_env):
    env.reset()
    if render_flag:
        env.render()
        time.sleep(0.01)

    total_reward = 0
    num_steps = 0

    q = [env.q]
    dq = [env.dq]
    u = []

    while True:
        action = controllers.calc_lqr_input(env, sim_env)
        u.append(np.copy(action))
        _, reward, is_terminal, _ = env._step(action)
        q.append(np.copy(env.q))
        dq.append(np.copy(env.dq))
        if render_flag:
            env.render()
            # time.sleep(0.00001)

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break
    return q, dq, u, total_reward, num_steps

# ENV_NAME = 'TwoLinkArm-v0'
ENV_NAME = 'TwoLinkArm-limited-torque-v0'
# ENV_NAME = 'TwoLinkArm-v1'
# ENV_NAME = 'TwoLinkArm-limited-torque-v1'

env = gym.make(ENV_NAME)
sim_env = gym.make(ENV_NAME)
# env = gym.make('TwoLinkArm-limited-torque-v0')
render_flag = True
q, dq, u, total_reward, num_steps = run_lqr_controller(env, render_flag, sim_env)
# Plot
plotthem(q, dq, u, num_steps, total_reward)
print total_reward, num_steps

# if __name__ == '__main__':
#     main()
