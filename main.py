import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import warnings
warnings.filterwarnings("ignore")

class Agent:
    def __init__(self):
        # Initializing the cart-pole environment from OpenAI's gym.
        self._env              = gym.make('CartPole-v1') 
        
        # Getting the number of states and possible actions from the environment.
        self._states           = self._env.observation_space.shape[0]
        self._possible_actions = self._env.action_space.n
        
        # Setting the number of episodes and possible actions for the CartPole environment.
        self._episodes         = 20
        self._actions          = [0, 1]
        self._training_steps   = 75000

        # Initializing the neural network model and the DQN agent.
        self._model            = self.init_model()
        self._agent            = self.init_agent()

        # Compiling, training, testing the agent and then closing the environment.
        self.compile_agent()
        self.train_agent()
        self.save_agent()
        self.test_agent()
        self.close_env()


    def compile_agent(self):
        # Defining the optimizer and metrics and then compiling the agent.
        opt = Adam(lr = 0.001)
        met = ["mae"]
        self._agent.compile(optimizer=opt, metrics=met)
        
    def train_agent(self):
        # Training the agent using the environment.
        self._agent.fit(self._env, nb_steps = self._training_steps, visualize = False, verbose = True)

    def test_agent(self):
        # Testing the agent's performance in the environment.
        self._agent.test(self._env, nb_episodes = self._episodes, visualize = True)

    def close_env(self):
        # Closing the cart-pole environment.
        self._env.close() 
   
    def init_model(self):
        # Creating a simple neural network model using Keras with three layers.
        model = Sequential()
        model.add(Flatten(input_shape=(1, self._states)))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self._possible_actions, activation="linear"))
        return model
    
    def init_agent(self):
        # Initializing the DQN agent with the model, memory, policy, and other parameters.
        agent = DQNAgent(
            model      = self._model,
            memory     = SequentialMemory(limit=50000, window_length=1),
            policy     = BoltzmannQPolicy(),
            nb_actions = self._possible_actions,
            nb_steps_warmup = 10,
            target_model_update=0.01,
        )
        return agent

    def save_agent(self):
        # Save the weights of the agent's model to a file.
        folder_name = "trained_models/" 
        filename = "cart_pole_episodes_" + str(self._episodes) + "_steps_" + str(self._training_steps) + ".h5"
        filepath = folder_name + filename 
        self._agent.save_weights(filepath, overwrite=True)

    def load_agent(self, filepath):
        # Load the weights for the agent's model from a file.
        # Arg filepath (str): Path to the file from which the weights should be loaded.
        self._agent.model.load_weights(filepath)

if __name__ == "__main__":
    # Executing the Agent class when the script runs.
    agent = Agent()