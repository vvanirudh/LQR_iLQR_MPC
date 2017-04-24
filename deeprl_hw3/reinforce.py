import gym
import numpy as np
import imitation
import tensorflow as tf
import keras
from keras import backend as K


def main():
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    
    # define model
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-08)
    model = imitation.load_model('../CartPole-v0_config.yaml')
    # model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')

    # set up gradients
    action_idx_tf = tf.placeholder(tf.int32, name="action_dx_tf")
    gradients = tf.gradients(tf.log(model.output[0][action_idx_tf]), model.trainable_weights)
    # gradients = tf.gradients(tf.log(model.output), model.trainable_weights)
    update_op = optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    # args
    args = {
        'num_episodes': 1e5,
        'max_episode_length': 1e5,
        'eval_freq': 100,
        'alpha': 1e-4,
        'gamma': 0.99,
        'num_eval_episodes': 100
    }

    def get_total_reward(model):
        """compute total reward

        Parameters
        ----------
        env: gym.core.Env
          The environment. 
        model: (your action model, which can be anything)

        Returns
        -------
        total_reward: float
        """
        return 0.0


    def choose_action(obs, num_actions=2, train=True):
        """choose the action 

        Parameters
        ----------
        model: (your action model, which can be anything)
        observation: given observation

        Returns
        -------
        p: float 
            probability of action 1
        action: int
            the action you choose
        """

        output = np.squeeze(model.predict(obs))
        # print("output")
        # print(output)
        if train:
            act_idx = np.random.choice(num_actions, p=output)
        else:
            act_idx = np.argmax(output)      

        return act_idx

    def evaluate():
        obs = eval_env.reset()
        obs = np.reshape(obs, (1,4))
        total_reward = []
        reward = 0
        for i in range(0, args['num_eval_episodes']):
            t = 0
            while t < args['max_episode_length']:
                action = choose_action(obs, train=False)
                obs, r, done, info = eval_env.step(action)
                obs = np.reshape(obs, (1,4))
                reward += r
                if done:
                    obs = eval_env.reset()
                    obs = np.reshape(obs, (1,4))
                    total_reward.append(reward)
                    reward = 0
                    break


        total_reward = np.array(total_reward)

        return total_reward.mean(), total_reward.min(), total_reward.max()

    def reinforce():
        """Policy gradient algorithm

        Parameters
        ----------
        env: your environment

        Returns
        -------
        total_reward: float
        """

        obs = env.reset()
        obs = np.reshape(obs, (1,4))
        episode = 1
        rewards = []
        obs_hist = []
        act_hist = []
        obs_hist.append(obs)
        t = 0

        # evaluate for the first time
        avg_r, min_r, max_r = evaluate()
        print("Episode, avg, min, max: %d, %f, %f, %f" % (episode,
            avg_r, min_r, max_r))

        while episode <= args['num_episodes']:
            t += 1
            action = choose_action(obs)
            obs, reward, done, info = env.step(action)
            obs = np.reshape(obs, (1,4))
            rewards.append(reward)
            obs_hist.append(obs)
            act_hist.append(action)
            if done or t > args['max_episode_length']:
                episode += 1
                t = 0

                # Calculate discounted return G
                G = []
                R = 0
                for r in reversed(rewards):
                    R = r + args['gamma'] * R
                    G.insert(0, R)

                # normalize G
                G = np.array(G)
                # G = (G - G.mean()) / (G.std() + np.finfo(np.float32).eps)

                # compute scaled gradients and update network weights
                num_obs = len(obs_hist)
                for i in range(0, num_obs-1):
                    grad = sess.run(gradients, feed_dict={
                        model.input: obs_hist[i],
                        action_idx_tf: act_hist[i]
                    })

                    # scale all gradients
                    scaled_grad = []
                    max_grad = 0
                    for g in grad:
                        # negate the gradient to perform gradient ascent
                        if max_grad < np.absolute(g).max():
                            max_grad = np.absolute(g).max()
                        scaled_grad.append(-G[i] * g)
                    # print("Max grad: %lf", max_grad)

                    # sess.run(update_op, feed_dict={gradients: scaled_grad})
                    sess.run(update_op, feed_dict={g: s for g, s in zip(gradients, scaled_grad)})

                # clear reward and observation history
                rewards = []
                obs_hist = []
                act_hist = []
                obs = env.reset()
                obs = np.reshape(obs, (1,4))
                obs_hist.append(obs)

                # check if we should evaluate now
                if episode % args['eval_freq'] == 0:
                    avg_r, min_r, max_r = evaluate()
                    print("Episode, avg, min, max: %d, %f, %f, %f" % (episode,
                        avg_r, min_r, max_r))

        return 0

    total_reward = reinforce()  

if __name__ == '__main__':
    main()
