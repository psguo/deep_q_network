#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import gym
import sys
import argparse
import matplotlib.pyplot as plt


class QNetwork:

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, sess, net_type, n_actions, n_features, lr, gamma, replace_iter):
        # Define your network architecture here. It is also a good idea to define any training operations
        # and optimizers here, initialize your variables, or alternately compile your model here.
        self.sess = sess
        self.replace_iter = replace_iter
        self.train_iter_counter = 0
        self.n_features = n_features
        self.n_actions = n_actions

        # build net
        if isinstance(n_features, int):
            self.s = tf.placeholder(tf.float32, [None, n_features], name='s')
            self.s_ = tf.placeholder(tf.float32, [None, n_features], name='s_')
        else:
            self.s = tf.placeholder(tf.float32, [None] + list(n_features), name='s')
            self.s_ = tf.placeholder(tf.float32, [None] + list(n_features), name='s_')

        self.r = tf.placeholder(tf.float32, [None, ], name='r')
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        self.done = tf.placeholder(tf.int32, [None, ], name='done')

        if net_type == 'linear':
            # current net
            with tf.variable_scope('q_net_cur'):
                self.q_cur = tf.layers.dense(inputs=self.s, units=n_actions, name='c1')

            # old net (using old fixed parameters)
            with tf.variable_scope('q_net_old'):
                q_old = tf.layers.dense(inputs=self.s_, units=n_actions, name='o1')

        # if net_type == 'DQN':
        #     # current net
        #     with tf.variable_scope('q_net_cur'):
        #         c1 = tf.layers.dense(inputs=self.s, units=64, activation=tf.nn.relu, name='c1')
        #         self.q_cur = tf.layers.dense(inputs=c1, units=n_actions, name='c2')

        #     # old net (using old fixed parameters)
        #     with tf.variable_scope('q_net_old'):
        #         o1 = tf.layers.dense(inputs=self.s_, units=64, activation=tf.nn.relu, name='o1')
        #         q_old = tf.layers.dense(inputs=o1, units=n_actions, name='o2')

        if net_type == 'DQN':
            # current net
            with tf.variable_scope('q_net_cur'):
                c1 = tf.layers.dense(inputs=self.s, units=16, activation=tf.nn.relu, name='c1')
                c2 = tf.layers.dense(inputs=c1, units=32, activation=tf.nn.relu, name='c2')
                self.q_cur = tf.layers.dense(inputs=c2, units=n_actions, name='c3')

            # old net (using old fixed parameters)
            with tf.variable_scope('q_net_old'):
                o1 = tf.layers.dense(inputs=self.s_, units=16, activation=tf.nn.relu, name='o1')
                o2 = tf.layers.dense(inputs=o1, units=32, activation=tf.nn.relu, name='o2')
                q_old = tf.layers.dense(inputs=o2, units=n_actions, name='o3')

        if net_type == 'Dueling':
            # current net
            with tf.variable_scope('q_net_cur'):
                c1 = tf.layers.dense(inputs=self.s, units=16, activation=tf.nn.relu, name='c1')
                c2 = tf.layers.dense(inputs=c1, units=32, activation=tf.nn.relu, name='c2')
                cv = tf.layers.dense(inputs=c2, units=1, name='cv')
                ca = tf.layers.dense(inputs=c2, units=n_actions, name='ca')
                self.q_cur = cv + (ca - tf.reduce_mean(ca, axis=1, keep_dims=True))

            # old net (using old fixed parameters)
            with tf.variable_scope('q_net_old'):
                o1 = tf.layers.dense(inputs=self.s_, units=16, activation=tf.nn.relu, name='o1')
                o2 = tf.layers.dense(inputs=o1, units=32, activation=tf.nn.relu, name='o2')
                ov = tf.layers.dense(inputs=o2, units=1, name='ov')
                oa = tf.layers.dense(inputs=o2, units=n_actions, name='oa')
                q_old = ov + (oa - tf.reduce_mean(oa, axis=1, keep_dims=True))

        if net_type == 'DQN_conv':
            # current net
            with tf.variable_scope('q_net_cur'):
                conv1_cur = tf.layers.conv2d(
                    inputs=self.s,
                    filters=16,
                    kernel_size=8,
                    strides=4,
                    padding='SAME',
                    activation=tf.nn.relu,
                    name='conv1_cur')
                conv2_cur = tf.layers.conv2d(
                    inputs=conv1_cur,
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    padding='SAME',
                    activation=tf.nn.relu,
                    name='conv2_cur')
                fc1_cur = tf.contrib.layers.flatten(conv2_cur)
                dense1_cur = tf.layers.dense(inputs=fc1_cur, units=256, activation=tf.nn.relu, name='dense1_cur')
                self.q_cur = tf.layers.dense(inputs=dense1_cur, units=n_actions, activation=tf.nn.relu,
                                             name='dense2_cur')

            # old net (using old fixed parameters)
            with tf.variable_scope('q_net_old'):
                conv1_old = tf.layers.conv2d(
                    inputs=self.s,
                    filters=16,
                    kernel_size=8,
                    strides=4,
                    padding='SAME',
                    activation=tf.nn.relu,
                    name='conv1_old')
                conv2_old = tf.layers.conv2d(
                    inputs=conv1_old,
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    padding='SAME',
                    activation=tf.nn.relu,
                    name='conv2_old')
                fc1_old = tf.contrib.layers.flatten(conv2_old)
                dense1_old = tf.layers.dense(inputs=fc1_old, units=256, activation=tf.nn.relu, name='dense1_old')
                q_old = tf.layers.dense(inputs=dense1_old, units=n_actions, activation=tf.nn.relu, name='dense2_old')

        with tf.variable_scope('q_target'):
            q_target = tf.stop_gradient(
                self.r + gamma * tf.reduce_max(q_old, axis=1, name='q_next_max') * tf.to_float(1 - self.done))

        with tf.variable_scope('q_cur_action'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            q_cur_action = tf.gather_nd(params=self.q_cur, indices=a_indices)

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.squared_difference(q_target, q_cur_action))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        cur_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net_cur')
        old_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_net_old')
        with tf.variable_scope('replace_old_net'):
            self.replace_op = [tf.assign(o, c) for o, c in zip(old_net_params, cur_net_params)]

        self.sess.run(tf.global_variables_initializer())

    def replace_old_net(self):
        self.sess.run(self.replace_op)

    def train(self, batch_memory_new):
        if self.train_iter_counter % self.replace_iter == 0:
            self.replace_old_net()

        # print("-------s-------")
        # print(batch_memory[:, :self.n_features])
        # print(batch_memory_new[0])
        # print("--------------------------")
        #
        # print("-------a-------")
        # print(batch_memory[:, self.n_features])
        # print(batch_memory_new[1])
        # print("--------------------------")
        #
        # print("-------r-------")
        # print(batch_memory[:, self.n_features + 1])
        # print(batch_memory_new[2])
        # print("--------------------------")
        #
        # print("-------s_-------")
        # print(batch_memory[:, self.n_features + 2:-1])
        # print(batch_memory_new[3])
        # print("--------------------------")
        #
        # print("-------done-------")
        # print(batch_memory[:, -1])
        # print(batch_memory_new[4])
        # print("--------------------------")

        self.sess.run(
            self.train_op,
            feed_dict={
                # self.s: batch_memory[:, :self.n_features],
                # self.a: batch_memory[:, self.n_features],
                # self.r: batch_memory[:, self.n_features + 1],
                # self.s_: batch_memory[:, self.n_features + 2:-1],
                # self.done: batch_memory[:, -1]
                self.s: batch_memory_new[0],
                self.a: batch_memory_new[1],
                self.r: batch_memory_new[2],
                self.s_: batch_memory_new[3],
                self.done: batch_memory_new[4]
            })

        self.train_iter_counter += 1

    def predict(self, observations):
        #TODO: parse observations using former experience
        return self.sess.run(self.q_cur, feed_dict={self.s: observations})

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        pass


class Replay_Memory:

    def __init__(self, n_features, memory_size=50000, burn_in=10000, history_length=1):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in transitions define the number of transitions that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.
        self.memory_size = memory_size
        self.burn_in = burn_in
        # self.memory = np.zeros((self.memory_size, n_features*2+3))

        if isinstance(n_features, int):
            self.obs = np.zeros([self.memory_size] + [n_features])
            self.obs_ = np.zeros([self.memory_size] + [n_features])
        else:
            self.obs = np.zeros([self.memory_size] + list(n_features), dtype=np.uint8)
            self.obs_ = np.zeros([self.memory_size] + list(n_features), dtype=np.uint8)

        self.action = np.zeros([self.memory_size])
        self.reward = np.zeros([self.memory_size])
        self.done = np.zeros([self.memory_size])

        self.memory_counter = 0
        self.history_length = history_length

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        sample_index_history_list = []
        sample_index = None
        if self.memory_counter > self.memory_size:
            if self.history_length > 1:
                append_index = self.memory_counter % self.memory_size
                sample_range = [i for i in range(self.memory_size) if
                                i < append_index or i > append_index + self.history_length - 1]
                sample_index = np.random.choice(sample_range, size=batch_size)
                sample_index_history_list = []
                for back in range(self.history_length):
                    sample_index_elem = sample_index - back
                    sample_index_elem[sample_index_elem < 0] = self.memory_size + sample_index_elem[
                        sample_index_elem < 0]
                    sample_index_history_list.append(sample_index_elem)
            else:
                sample_index = np.random.choice(self.memory_size, size=batch_size)
        else:
            if self.history_length > 1:
                sample_range = [i for i in range(self.memory_counter) if i > self.history_length - 1]
                sample_index = np.random.choice(sample_range, size=batch_size)
                sample_index_history_list = []
                for back in range(self.history_length):
                    sample_index_elem = sample_index - back
                    sample_index_history_list.append(sample_index_elem)
            else:
                sample_index = np.random.choice(self.memory_counter, size=batch_size)

        if self.history_length > 1:
            history_obs = np.zeros([batch_size] + self.obs.shape[1:] + [self.history_length])
            history_obs_ = np.zeros([batch_size] + self.obs.shape[1:] + [self.history_length])

            for i, sample_index_elem in enumerate(sample_index_history_list):
                history_obs[:, :, :, i] = self.obs[sample_index_elem]
                history_obs_[:, :, :, i] = self.obs_[sample_index_elem]
            batch_new = (
                history_obs, self.action[sample_index], self.reward[sample_index], history_obs_,
                self.done[sample_index])
        else:
            batch_new = (
            self.obs[sample_index], self.action[sample_index], self.reward[sample_index], self.obs_[sample_index],
            self.done[sample_index])

        # batch = self.memory[sample_index]
        return batch_new

    def append(self, s, a, r, s_, done):
        # Appends transition to the memory.
        # transition = np.hstack((s, a, r, s_, done))
        append_index = self.memory_counter % self.memory_size
        # self.memory[append_index, :] = transition

        if len(s.shape) == 3:
            self.obs[append_index, :] = self.convert_frame(s)
            self.obs_[append_index, :] = self.convert_frame(s_)
        else:
            self.obs[append_index, :] = s
            self.obs_[append_index, :] = s_

        self.reward[append_index] = r
        self.action[append_index] = a
        self.done[append_index] = done

        self.memory_counter += 1

    def convert_frame(self, frame):
        import cv2
        frame = cv2.cvtColor(cv2.resize(frame, (84, 110)), cv2.COLOR_BGR2GRAY)
        frame = frame[26:110, :]
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)

        # # Change to grayscale
        # frame = np.mean(frame, axis=2).astype(np.uint8)
        # res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
        # # Downsampling
        # frame = frame[::2, ::2]
        return frame

    def get_previous_obs(self, n):
        history_obs = np.zeros(list(self.obs.shape[1:]) + [n])
        current_idx = self.memory_counter % self.memory_size
        for i in range(1, n+1):
            history_obs[:,:,i-1] = self.obs[current_idx-i if current_idx-i >= 0 else self.memory_size+current_idx-i, :, :]
        return history_obs

class DQN_Agent:

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env, test_env, sess, render, net_type, lr, gamma, e_start, e_end, e_decay, replace_iter,
                 memory_size, burn_in, batch_size, n_episode, test_iter, test_interval, video_dir,
                 is_image_input=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env = env
        self.render = render
        self.epsilon = e_start
        self.e_end = e_end
        self.e_decay = e_decay
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.n_episode = n_episode
        self.test_iter = test_iter
        self.test_interval = test_interval
        self.record_episode = 0
        self.is_image_input = is_image_input
        if video_dir is not None:
            self.test_env = gym.wrappers.Monitor(test_env, video_dir, force=True, video_callable=self.record)
        else:
            self.test_env = test_env

        if not is_image_input:
            self.Q_network = QNetwork(
                sess=sess,
                net_type=net_type,
                n_actions=self.env.action_space.n,
                n_features=self.env.observation_space.shape[0],
                lr=lr,
                gamma=gamma,
                replace_iter=replace_iter
            )

            self.memory = Replay_Memory(n_features=self.env.observation_space.shape[0], memory_size=memory_size,
                                        burn_in=self.burn_in)
        else:
            self.Q_network = QNetwork(
                sess=sess,
                net_type=net_type,
                n_actions=self.env.action_space.n,
                n_features=[84, 84, 4],
                lr=lr,
                gamma=gamma,
                replace_iter=replace_iter
            )

            self.memory = Replay_Memory(n_features=[84, 84], memory_size=memory_size, burn_in=self.burn_in, history_length=4)

    def epsilon_greedy_policy(self, observation, epsilon):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() > epsilon:
            action_values = self.Q_network.predict(np.expand_dims(observation, axis=0))[0]
            action = np.argmax(action_values)
        else:
            action = np.random.randint(0, self.env.action_space.n)
        return action

    def greedy_policy(self, observation):
        # Creating greedy policy for test time. 
        action_values = self.Q_network.predict(np.expand_dims(observation, axis=0))[0]
        action = np.argmax(action_values)
        return action

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        if self.burn_in > 0:
            self.burn_in_memory()

        print('Start training...')
        self.performance_hist = []
        self.episode = 0

        for self.episode in range(self.n_episode):
            observation = self.env.reset()
            total_reward = 0

            # eps_step = 0
            while True:
                if self.render:
                    self.env.render()

                if self.Q_network.train_iter_counter % self.test_interval == 0:
                    performance, _ = self.test(self.test_iter, self.episode)
                    self.performance_hist.append([self.episode, self.Q_network.train_iter_counter, performance])

                if self.is_image_input:
                    obs_history = self.memory.get_previous_obs(self.memory.history_length - 1)
                    total_obs = np.concatenate((np.expand_dims(self.memory.convert_frame(observation), axis=2), obs_history), axis=2)
                    action = self.epsilon_greedy_policy(total_obs, self.epsilon)
                else:
                    action = self.epsilon_greedy_policy(observation, self.epsilon)
                observation_, reward, done, info = self.env.step(action)
                # eps_step += 1
                # true_done = done
                # if eps_step == 199:
                #     true_done = False
                # self.memory.append(observation, action, reward, observation_, true_done)
                self.memory.append(observation, action, reward, observation_, done)

                batch_memory_new = self.memory.sample_batch(batch_size=self.batch_size)
                self.Q_network.train(batch_memory_new)

                total_reward += reward
                observation = observation_

                if self.epsilon > self.e_end:
                    self.epsilon -= self.e_decay

                if done:
                    # print('episode: ', self.episode, 'total reward: ', round(total_reward, 2), 'epsilon', round(self.epsilon, 2))
                    break

        print('Training done. Evaluting final model...')
        performance, _ = self.test(100, self.episode)
        self.performance_hist.append([self.episode, self.Q_network.train_iter_counter, performance])

    def test(self, test_iter, episode_n, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        reward_rec = np.zeros(test_iter)
        for episode in range(test_iter):
            observation = self.test_env.reset()
            observations = np.zeros([84, 84] + [self.memory.history_length])
            counter = 0
            while True:
                if self.render:
                    self.test_env.render()
                # self.test_env.render()

                if self.is_image_input:
                    observations[:,:, counter%self.memory.history_length] = self.memory.convert_frame(observation)
                    action = self.epsilon_greedy_policy(observations, 0.05)
                else:
                    action = self.epsilon_greedy_policy(observation, 0.05)
                observation_, reward, done, info = self.test_env.step(action)
                reward_rec[episode] += reward
                observation = observation_

                if done:
                    break
                counter += 1
        avg = np.average(reward_rec)
        std = np.std(reward_rec)
        print('---------------Testing---------------')
        print('Episode:     ', episode_n)
        print('Iteration:   ', self.Q_network.train_iter_counter)
        print('Avg reward:  ', avg)
        print('std:         ', std)

        return avg, std

    def plot_performance(self):
        performance_hist = np.array(self.performance_hist)
        plt.plot(performance_hist[:, 1], performance_hist[:, 2])
        plt.ylabel('Average performance')
        plt.xlabel('Iteration')
        plt.show()

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        print('Burning in memory...')
        while self.memory.memory_counter < self.memory.burn_in:
            observation = self.env.reset()

            while True:
                if self.render:
                    self.env.render()

                action = np.random.randint(0, self.env.action_space.n)
                observation_, reward, done, info = self.env.step(action)
                self.memory.append(observation, action, reward, observation_, done)

                if done:
                    break

    def record(self, e):

        if self.episode + 1 >= self.record_episode:
            # print(self.episode, ' ', self.record_episode)
            self.record_episode += int(self.n_episode / 3)
            return True
        else:
            return False


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env_name', type=str)
    parser.add_argument('--render', dest='render', type=int, default=1)
    # parser.add_argument('--train', dest='train', type=int, default=1)
    # parser.add_argument('--model', dest='model_file', type=str)
    parser.add_argument('--net_type', dest='net_type', type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9)
    parser.add_argument('--e_start', dest='e_start', type=float, default=0.5)
    parser.add_argument('--e_end', dest='e_end', type=float, default=0.05)
    parser.add_argument('--e_decay', dest='e_decay', type=float, default=1e-5)
    parser.add_argument('--replace_iter', dest='replace_iter', type=int, default=200)
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=50000)
    parser.add_argument('--burn_in', dest='burn_in', type=int, default=10000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--n_episode', dest='n_episode', type=int, default=100)
    parser.add_argument('--test_iter', dest='test_iter', type=int, default=20)
    parser.add_argument('--test_interval', dest='test_interval', type=int, default=10000)
    parser.add_argument('--video_dir', dest='video_dir', type=str, default=None)
    return parser.parse_args()


def record(episode):
    if episode % 10 == 0:
        return True
    else:
        return False


def main(args):
    args = parse_arguments()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    # np.random.seed(2)
    # tf.set_random_seed(2)
    # env.seed(1)
    # test_env.seed(1)

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 

    if args.env_name == 'SpaceInvaders-v0':
        agent = DQN_Agent(env=env, test_env=test_env, sess=sess,
                          render=args.render, net_type=args.net_type,
                          lr=args.lr, gamma=args.gamma,
                          e_start=args.e_start, e_end=args.e_end,
                          e_decay=args.e_decay, replace_iter=args.replace_iter,
                          memory_size=args.memory_size, burn_in=args.burn_in,
                          batch_size=args.batch_size, n_episode=args.n_episode,
                          test_iter=args.test_iter,
                          test_interval=args.test_interval, video_dir=args.video_dir,
                          is_image_input=True)
    else:
        agent = DQN_Agent(env=env, test_env=test_env, sess=sess,
                          render=args.render, net_type=args.net_type,
                          lr=args.lr, gamma=args.gamma,
                          e_start=args.e_start, e_end=args.e_end,
                          e_decay=args.e_decay, replace_iter=args.replace_iter,
                          memory_size=args.memory_size, burn_in=args.burn_in,
                          batch_size=args.batch_size, n_episode=args.n_episode,
                          test_iter=args.test_iter,
                          test_interval=args.test_interval, video_dir=args.video_dir)

    agent.train()
    agent.plot_performance()


if __name__ == '__main__':
    main(sys.argv)

