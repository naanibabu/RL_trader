import numpy as np
from replay_buffer import ReplayBuffer
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

def mlp(input_dim, n_action, n_hidden_layers=1, hidden_dim=32):
  """ A multi-layer perceptron """

  # input layer
  i = Input(shape=(input_dim,))
  x = i

  # hidden layers
  for _ in range(n_hidden_layers):
    x = Dense(hidden_dim, activation='relu')(x)
  
  # final layer
  x = Dense(n_action)(x)

  # make the model
  model = Model(i, x)

  model.compile(loss='mse', optimizer='adam')
  print((model.summary()))
  return model

class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = ReplayBuffer(state_size, action_size, size=500)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = mlp(state_size, action_size)


  def update_replay_memory(self, state, action, reward, next_state, done):
    self.memory.store(state, action, reward, next_state, done)


  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = self.model(state, training=False)
    return np.argmax(act_values[0])  # returns action

  @tf.function
  def replay(self, batch_size=32):
    # first check if replay buffer contains enough data
    if self.memory.size < batch_size:
      return

    # sample a batch of data from the replay memory
    minibatch = self.memory.sample_batch(batch_size)
    states = minibatch['s']
    actions = minibatch['a']
    rewards = minibatch['r']
    next_states = minibatch['s2']
    done = minibatch['d']

    # Calculate the tentative target: Q(s',a)
    target = rewards + (1 - done) * self.gamma * np.amax(self.model.predict(next_states, verbose=0), axis=1)

    # With the Keras API, the target (usually) must have the same
    # shape as the predictions.
    # However, we only need to update the network for the actions
    # which were actually taken.
    # We can accomplish this by setting the target to be equal to
    # the prediction for all values.
    # Then, only change the targets for the actions taken.
    # Q(s,a)
    target_full = self.model.predict(states, verbose=0)
    target_full[np.arange(batch_size), actions] = target

    # Run one training step
    self.model.train_on_batch(states, target_full)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)
