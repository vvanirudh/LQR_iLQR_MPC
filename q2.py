from keras.models import model_from_yaml
from keras.callbacks import Callback
import gym
import numpy as np
from deeprl_hw3 import imitation

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.info = []

    def on_batch_end(self, batch, logs={}):
        self.info.append((logs.get('loss'), logs.get('acc')))

def generate_training_data(env, expert, num_episodes):
    return imitation.generate_expert_training_data(expert, env, num_episodes, render=False)

def train_model(model_config_yaml, train_data, epochs):
    model = model_from_yaml(model_config_yaml)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    history = LossHistory()
    model.fit(train_data[0], train_data[1], batch_size=32, epochs=epochs, callbacks=[history], verbose=False)
    return model, history.info[-1]

def find_nearest_distance(array, value):
    min_dist = 10000000
    for i in xrange(len(array)):
        dist = np.linalg.norm(array[j]-value)
        if dist < min_dist:
            min_dist = dist
    return min_dist
env = gym.make('CartPole-v0')
wrapped_env = imitation.wrap_cartpole(env)
expert = imitation.load_model('CartPole-v0_config.yaml', 'CartPole-v0_weights.h5f')

model_config_path = 'CartPole-v0_config.yaml'
with open(model_config_path, 'r') as f:
        model_config_yaml = f.read()

num_epochs = 50
# Behaviour cloning experiments
expts = {}
# Expts with expert policy
expt_name = 'expert'
_, mean_rewards_env, std_rewards_env = imitation.test_cloned_policy(env, expert, render=False)
_, mean_rewards_wrapped_env, std_rewards_wrapped_env = imitation.test_cloned_policy(wrapped_env, expert, render=False)
expts[expt_name] = {'loss': 0, 'acc': 1, 'mean_rewards_env': mean_rewards_env,
                    'std_rewards_env': std_rewards_env, 'mean_rewards_wrapped_env': mean_rewards_wrapped_env,
                    'std_rewards_wrapped_env': std_rewards_wrapped_env}
# Expts with cloning policy
for num_eps in [1,10,50,100]:
    expt_name = "clone_policy_%deps"%num_eps
    train_data = generate_training_data(env, expert, num_eps)
    cloned_policy, final_info = train_model(model_config_yaml, train_data, num_epochs)
    _, mean_rewards_env, std_rewards_env = imitation.test_cloned_policy(env, cloned_policy, render=False)
    _, mean_rewards_wrapped_env, std_rewards_wrapped_env = imitation.test_cloned_policy(wrapped_env, cloned_policy, render=False)
    expts[expt_name] = {'loss': final_info[0], 'acc': final_info[1], 'mean_rewards_env': mean_rewards_env,
                        'std_rewards_env': std_rewards_env, 'mean_rewards_wrapped_env': mean_rewards_wrapped_env,
                        'std_rewards_wrapped_env': std_rewards_wrapped_env}

print "Expt Name, Loss, Accuracy, Mean_Reward_Env, Std_Rewards_Env, Mean_Rewards_Wrapped_Env, Std_Rewards_Wrapped_Env"
for expt in expts:
    res = expts[expt]
    print "%s %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f"%(expt, res['loss'], res['acc'], res['mean_rewards_env'], res['std_rewards_env'], res['mean_rewards_wrapped_env'], res['std_rewards_wrapped_env'])

# DAGGER
k_eps = 5
threshold = 0.0001
train_data = generate_training_data(env, expert, k_eps)
results = []
for i in xrange(30):
    cloned_policy, final_info = train_model(model_config_yaml, train_data, num_epochs)
    new_states, _ = generate_training_data(env, cloned_policy, k_eps)
    additional_states = []
    additional_actions = []
    for j in xrange(len(new_states)):
        state = new_states[j]
        if find_nearest_distance(train_data[0], state) >= threshold:
            action = np.argmax(
                expert.predict_on_batch(state[np.newaxis, ...])[0])
            additional_states.append(state)
            additional_actions.append(action)
    onehot_actions = np.eye(2)[additional_actions]
    additional_actions = np.array(onehot_actions) 
    train_data = (np.vstack((train_data[0],additional_states)), np.vstack((train_data[1],additional_actions)))
    
    total_rewards_env, mean_rewards_env, std_rewards_env = imitation.test_cloned_policy(env, cloned_policy, render=False)
    total_rewards_wrapped_env, mean_rewards_wrapped_env, std_rewards_wrapped_env = imitation.test_cloned_policy(wrapped_env, cloned_policy, render=False) 
    results.append([i+1, min(total_rewards_env), max(total_rewards_env), mean_rewards_env, std_rewards_env,
                    min(total_rewards_wrapped_env), max(total_rewards_wrapped_env), mean_rewards_wrapped_env, std_rewards_wrapped_env])

for result in results:
    print result
    
