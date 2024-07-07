import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import tensorflow as tf
import tensorflow.keras.layers as tfl
import v0_warehouse_robot_env


# Memory for experience replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class WarehouseDQN:
    # Hyperparameters
    learning_rate = 0.001
    mini_batch_size = 128  # How often do we optimize the network
    gamma = 0.99
    target_update_freq = 128  # How often do we copy
    replay_memory_size = 1000  # Size of the replay memory
    epsilon_end = 0.1
    epsilon_decay = 0.99

    def __init__(self) -> None:
        self.memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

    def create_q_model(self, input_dim, output_dim):
        model = tf.keras.Sequential()
        model.add(tfl.Dense(input_shape=[input_dim], units=64, activation="relu"))
        # model.add(tfl.Dense(64, activation="relu")),
        model.add(tfl.Dense(output_dim, activation="linear"))
        model.compile(optimizer=self.optimizer, loss=self.loss_function)
        return model

    def train(self, episodes, render=False):
        env = gym.make(
            "warehouse-robot-v0",
            render_mode="human" if render else None,
        )
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        policy_dqn = self.create_q_model(self.num_states, self.num_actions)
        target_dqn = self.create_q_model(self.num_states, self.num_actions)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        epsilon = 1

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        # Track the number of steps for the robot to find the target.
        steps_per_episode = np.zeros(episodes)

        print(f"Initialization done, with {episodes} episodes")

        for i in range(episodes):
            print(f"Episode {i}, epsilon {epsilon}")
            state = env.reset()[0]  # Initialize to state 0
            # print(state)

            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions
            steps_to_complete = 0
            j = 0
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and not truncated:
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = (
                        env.action_space.sample()
                    )  # actions: 0=left,1=down,2=right,3=up
                else:
                    # Select best action
                    # print(f"STate: {state}")
                    # print(
                    #     f"Reshaped: {tf.convert_to_tensor(state.reshape(1,6), dtype=tf.float32)}"
                    # )
                    # state_dqn_input = self.state_to_dqn_input(state, self.num_states)
                    # print(f"State dqn input: {state_dqn_input}")
                    # print(
                    #     f"Tensor rep: {tf.convert_to_tensor([state_dqn_input], dtype=tf.float32)}"
                    # )
                    # q_values = policy_dqn(
                    #     tf.convert_to_tensor([state_dqn_input], dtype=tf.float32),
                    #     training=False,
                    # )
                    # q_values = policy_dqn(
                    #     tf.convert_to_tensor(state.reshape(1, 6), dtype=tf.float32),
                    #     training=False,
                    # )
                    q_values = policy_dqn.predict(state.reshape(1, 6), verbose=0)
                    # print(q_values)
                    action = np.argmax(q_values[0])
                    print(f"Taking greedy action: {action} in state: {state}")

                # Execute action
                new_state, reward, terminated, truncated, _ = env.step(action)
                # print(f"Execute action: {action}, new_state:\n{new_state}")
                # if j > 25:
                #     return

                # j += 1

                # Save experience into memory
                self.memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

                steps_to_complete += 1

            # Keep track of the rewards collected per episode
            if reward == 1:
                print(f"Steps to complete the episode: {steps_to_complete}")
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if (
                len(self.memory) > self.mini_batch_size
                and np.sum(rewards_per_episode) > 0
            ):
                print(f"Optimizing, memory len = {len(self.memory)}")
                mini_batch = self.memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_end)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.target_update_freq:
                    print("Copying weights")
                    target_dqn.set_weights(policy_dqn.get_weights())
                    step_count = 0
        print(f"Episodes overn")
        # Close environment
        env.close()

        # Save policy
        policy_dqn.save("warehouse_dqn.keras")

        # Create new graph
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100) : (x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig("warehouse_dqn.png")

    def state_to_dqn_input(self, state: int, num_states: int):
        input_tensor = np.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, new_states, rewards, terminals = zip(*mini_batch)

        # states = np.vstack([self.state_to_dqn_input(s, self.num_states) for s in states])
        # new_states = np.vstack([self.state_to_dqn_input(ns, self.num_states) for ns in new_states])

        # states = tf.convert_to_tensor(
        #     np.vstack([self.state_to_dqn_input(s, self.num_states) for s in states]),
        #     dtype=tf.float32,
        # )
        states = tf.convert_to_tensor(np.vstack([s for s in states]))
        # new_states = tf.convert_to_tensor(
        #     np.vstack(
        #         [self.state_to_dqn_input(ns, self.num_states) for ns in new_states]
        #     ),
        #     dtype=tf.float32,
        # )
        new_states = tf.convert_to_tensor(
            np.vstack([ns for ns in new_states]), dtype=tf.float32
        )

        # future_qs = target_dqn(new_states, training=False).numpy().max(axis=1)
        # target_qs = np.where(terminals, rewards, rewards + self.gamma * future_qs)

        # masks = tf.one_hot(actions, policy_dqn.output_shape[1])
        future_qs = tf.reduce_max(target_dqn(new_states), axis=1)
        target_qs = rewards + self.gamma * future_qs * (
            1 - np.array(terminals, dtype=np.float32)
        )

        masks = tf.one_hot(actions, self.num_actions)

        with tf.GradientTape() as tape:
            q_values = policy_dqn(states)
            # print(f"Shape of q_values: {q_values.shape}")
            # print(f"Q values during optimization: {q_values}")
            q_action = tf.reduce_sum(q_values * masks, axis=1)
            loss = self.loss_function(target_qs, q_action)

        grads = tape.gradient(loss, policy_dqn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, policy_dqn.trainable_variables))

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes):
        # Create FrozenLake instance
        env = gym.make(
            "warehouse-robot-v0",
            render_mode="human",
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = tf.keras.models.load_model("warehouse_dqn.keras")

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False  # True when agent falls in hole or reached goal
            truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while not terminated and not truncated:
                # Select best action
                tensor = tf.convert_to_tensor(
                    [self.state_to_dqn_input(state, num_states)], dtype=tf.float32
                )

                action = tf.argmax(
                    policy_dqn(tensor)[0]
                ).numpy()  # TODO change here how we transform our state

                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()


if __name__ == "__main__":
    warehouse_bot = WarehouseDQN()
    warehouse_bot.train(500)
    # warehouse_bot.test(4)
