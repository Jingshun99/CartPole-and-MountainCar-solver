
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

train_history = {'episodes': [], 'reward': []}
test_history = {'episodes': [], 'reward': []}

def RLModel(input_shape, action_space):
    # Input with the shape of the observation space
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation='relu',kernel_initializer='he_normal')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation='relu', kernel_initializer='he_normal')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation='relu', kernel_initializer='he_normal')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation='linear', kernel_initializer='he_normal')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')

    # Compute loss with mean square error, Adam as optimizer and mean absolute error as performance metrics
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=['mae'])
    model.summary()
    return model

class DQNAgent:

    ## Initialise parameters
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.OBS_SIZE = self.env.observation_space.shape[0]
        self.NUM_ACTIONS = self.env.action_space.n
        self.EPISODES = 1000 # training episodes
        self.MEMORY = deque(maxlen=2000) # memory size
        self.GAMMA = 0.95    # discount rate of future reward
        self.EPSILON = 1.0  # exploration rate
        self.EPSILON_MIN = 0.001 # min exploration rate
        self.EPSILON_DECAY = 0.999 # exploration decay rate
        self.BATCH_SIZE = 64 # memory used for training
        self.TRAIN_START = 1000

        # creating model
        self.model = RLModel(input_shape=(self.OBS_SIZE,), action_space = self.NUM_ACTIONS)

    ## Append previous experience to memory and decay exploration rate
    def remember(self, state, action, reward, next_state, done):
        self.MEMORY.append((state, action, reward, next_state, done))
        if len(self.MEMORY) > self.TRAIN_START:
            if self.EPSILON > self.EPSILON_MIN:
                self.EPSILON *= self.EPSILON_DECAY

    ## Action chosen based on exploration or exploitation
    def act(self, state):
        if np.random.random() <= self.EPSILON:
            return random.randrange(self.NUM_ACTIONS)
        else:
            return np.argmax(self.model.predict(state))

    ## Train the Neural Network with previous experience
    def replay(self):
        if len(self.MEMORY) < self.TRAIN_START:
            return
        # Randomly sample batch of experiences from the memory
        batch = random.sample(self.MEMORY, min(len(self.MEMORY), self.BATCH_SIZE))

        state = np.zeros((self.BATCH_SIZE, self.OBS_SIZE))
        next_state = np.zeros((self.BATCH_SIZE, self.OBS_SIZE))
        action, reward, done = [], [], []

        # Extract info from batches
        for i in range(self.BATCH_SIZE):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state[i] = batch[i][3]
            done.append(batch[i][4])

        # Predict Q values
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        # Update Q values
        for i in range(self.BATCH_SIZE):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # DQN chooses the max Q value among next actions
                # Selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.GAMMA * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.BATCH_SIZE, verbose=0)

    ## Save the model
    def save(self, name):
        self.model.save(name)

    ## Load the model
    def load(self, name):
        self.model = load_model(name)

    ## Train the agent to learn weights 
    def train(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.OBS_SIZE])
            done = False
            i = 0
            while not done:
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.OBS_SIZE])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print('episode: {}, score: {}, e: {:.2}'.format(e+1, i, self.EPSILON))
                    train_history['episodes'].append(e+1)
                    train_history['reward'].append(i)
                    if i == 200:
                        #print('Saving weights as cartpole_DQN.h5')
                        #self.save('cartpole_DQN.h5')
                        return
                self.replay()

    ## Control cartpole using trained weights
    def test(self):
        #self.load('cartpole_DQN(BEST).h5')
        total_score = 0
        for e in range(100):
            state = self.env.reset()
            state = np.reshape(state, [1, self.OBS_SIZE])
            done = False
            i = 0
            while not done:
                #self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.OBS_SIZE])
                i += 1
                if done:
                    total_score += i
                    print('episode: {}, score: {}'.format(e+1, i))
                    test_history['episodes'].append(e+1)
                    test_history['reward'].append(i)
                    break
        print('Average score: {}'.format(total_score/100))
        if total_score/100 >= 195:
            print('Solved')
        else:
            print('Failed to solve')

if __name__ == '__main__':
    agent = DQNAgent()
    print('Training...')
    agent.train()
    print('Testing...')
    agent.test()

    ## Plot results
    plt.plot(train_history['episodes'],train_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Cartpole - Deep Q Learning (Training)')
    plt.show()
    plt.plot(test_history['episodes'],test_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Cartpole - Deep Q Learning (Testing)')
    plt.show()