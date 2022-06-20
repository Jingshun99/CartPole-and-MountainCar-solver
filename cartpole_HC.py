import gym
import numpy as np
import matplotlib.pyplot as plt

train_history = {'episodes': [], 'reward': []}
test_history = {'episodes': [], 'reward': []}

class HillClimbingAgent():

    ## Initialise environment
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.OBS_SIZE = self.env.observation_space.shape
        self.NUM_ACTIONS = self.env.action_space.n
        self.build_model()
        
    ## Initialise parameters
    def build_model(self):
        self.WEIGHTS = 0.0004*np.random.rand(*self.OBS_SIZE, self.NUM_ACTIONS) # random initialise weights
        self.EPSILON = 0.02 # exploration rate
        self.EPSILON_MIN = 0.0001 # minimum exploration rate
        self.EPSILON_MAX = 2 # maximum exploration rate
        self.best_REWARD = -np.Inf # best reward obtained
        self.best_WEIGHTS = np.copy(self.WEIGHTS) # best weight 
        
    ## Choose action that gives the highest value
    def get_action(self, state):
        p = np.dot(state, self.WEIGHTS)
        action = np.argmax(p)
        return action
    
    ## Update the weights
    def update_model(self, reward):
        if reward >= self.best_REWARD:
            self.best_REWARD = reward
            self.best_WEIGHTS = np.copy(self.WEIGHTS)
            self.EPSILON = max(self.EPSILON/2,self.EPSILON_MIN) # Decrease exploration rate
        else:
            self.EPSILON = min(self.EPSILON*2,self.EPSILON_MAX) # Increase exploration rate
            
        self.WEIGHTS = self.best_WEIGHTS + self.EPSILON * np.random.rand(*self.OBS_SIZE, self.NUM_ACTIONS)

    ## Train the agent
    def train(self):
        num_episodes = 100

        for ep in range(num_episodes):
            state = self.env.reset()
            i = 0
            done = False
            while not done:
                action = self.get_action(state)
                state, reward, done, info = self.env.step(action)
                #self.env.render()
                i += 1

            self.update_model(i)
            print("episode: {}, score: {}".format(ep+1, i))
            train_history['episodes'].append(ep+1)
            train_history['reward'].append(i)

    ## Test the agent with learned weights
    def test(self):
        num_episodes = 100
        total_score = 0

        for ep in range(num_episodes):
            state = self.env.reset()
            i = 0
            done = False
            while not done:
                action = self.get_action(state)
                state, reward, done, info = self.env.step(action)
                #self.env.render()
                i += 1
                if done:
                    total_score += i
                    print('episode: {}, score: {}'.format(ep+1, i))
                    test_history['episodes'].append(ep+1)
                    test_history['reward'].append(i)
                    break
        print('Average score: {}'.format(total_score/100))
        if total_score/100 >= 195:
            print('Solved')
        else:
            print('Failed to solve')


if __name__ == '__main__':
    agent = HillClimbingAgent()
    print('Training...')
    agent.train()
    print()
    print('Testing...')
    agent.test()

    ## Plot results
    plt.plot(train_history['episodes'],train_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Cartpole - Hill Climbing (Training)')
    plt.show()
    plt.plot(test_history['episodes'],test_history['reward'])
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Cartpole - Hill Climbing (Testing)')
    plt.show()

