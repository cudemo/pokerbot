import configparser
import time
from pathlib import Path
from random import randint

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras.layers import BatchNormalization
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from components.MemoryBuffer import MemoryBuffer


class DdqnModel(object):

    def __init__(self, name, alpha, gamma, input_layer_size, number_of_parameters, out_layer_size, memory_size=10000,
                 batch_size=32):

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        config.sections()

        enable_model_load = config['model_weights'].getboolean('enable_load_model_weights')
        self.enable_model_save = config['model_weights'].getboolean('enable_save_model_weights')
        self.tensorboard_visualization = config['tensorboard'].getboolean('enable_ddqn')

        # Hyperparameters
        self.memory = MemoryBuffer(max_size=memory_size, number_of_parameters=input_layer_size)
        self.gamma = gamma
        self.learning_rate = alpha
        self.batch_size = batch_size
        self.out_layer_size = out_layer_size
        self.replace_target_network_after = config['model_settings'].getint('ddqn_replace_network_interval')
        self.action_space = [i for i in range(out_layer_size)]

        loss = Huber()
        optimizer = Adam(learning_rate=alpha)

        # Epsilon Greedy Strategy
        self.epsilon = 1.0  # enable epsilon = 1.0 only when changing model, else learned weights from .h5 are used.
        self.epsilon_decay = 0.9985
        self.epsilon_min = 0.005

        # Keras Models
        hl1_dims = 128
        hl2_dims = 64
        hl3_dims = 64

        self.dqn_eval = self._build_model(hl1_dims, hl2_dims, hl3_dims, input_layer_size, out_layer_size, optimizer,
                                          loss)
        self.dqn_target = self._build_model(hl1_dims, hl2_dims, hl3_dims, input_layer_size, out_layer_size, optimizer,
                                            loss)

        if self.tensorboard_visualization:
            comment = 'adam-huber-reward_per'
            path = config['tensorboard']['file_path']
            tboard_name = '{}{}-cmt-{}_hl1_dims-{}_hl2_dims-{}-time-{}'.format(path, name, comment, hl1_dims, hl2_dims,
                                                                               int(time.time()))
            self.tensorboard = TensorBoard(tboard_name.format())

        self.keras_weights_filename = '{}.keras'.format(name)
        self.model_loaded = False
        if enable_model_load:
            self.load_model()
        else:
            print('Applying epsilon greedy strategy')

    def remember(self, state, action, reward, next_state, done, street):
        self.memory.store_state(state, action, reward, next_state, done)

    def replay(self):
        if not self.memory.mem_cntr > self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.get_sample_batch(self.batch_size)

        q_next = self.dqn_target.predict(new_states)
        q_eval = self.dqn_eval.predict(new_states)  # Estimate new next action target given the state
        q_pred = self.dqn_eval.predict(states)
        q_target = q_pred  # needed to calculate differences which shall be zero

        max_actions = np.argmax(q_eval, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = \
            rewards + self.gamma * q_next[batch_index, max_actions.astype(
                int)] * dones  # done has been inverted, see MemoryBuffer terminal_memory

        # Update active eval network with DDQN function using target network
        if self.tensorboard_visualization:
            self.dqn_eval.fit(states, q_target, verbose=0, epochs=1, callbacks=[self.tensorboard])
        else:
            self.dqn_eval.fit(states, q_target, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        if self.memory.mem_cntr % self.replace_target_network_after == 0:
            self._update_target_network_weights()

    def choose_action(self, state, street):

        if np.random.rand() <= self.epsilon:
            action = randint(0, self.out_layer_size - 1)
        else:
            size = state.shape
            state = np.reshape(state, [1, size[0]])
            q_values = self.dqn_eval.predict([state, np.ones(self.out_layer_size).reshape(1, self.out_layer_size)])
            action = np.argmax(q_values[0])

        return action

    def save_model(self):
        if not self.enable_model_save:
            return
        self.dqn_eval.save('./model/' + self.keras_weights_filename)
        print('Model weights have been saved to ./model/%f\n', self.keras_weights_filename)

    def load_model(self):
        model = Path('./model/' + self.keras_weights_filename)
        if model.is_file():
            try:
                self.dqn_eval.load_weights('./model/' + self.keras_weights_filename)
                self._update_target_network_weights()
                self.epsilon = 0.05
                self.model_loaded = True
                print('Pretrained model loaded, epsilon greedy set to 0.05')
            except ValueError:
                print("Model could not be loaded ... using epsilon greedy strategy")
        else:
            print('Applying epsilon greedy strategy')

    def reset_state(self):
        self.dqn_eval.reset_states()

    def _build_model(self, hl1_dims, hl2_dims, hl3_dims, input_layer_size, output_layer_size, optimizer, loss):
        model = Sequential()

        # The Layer Size is being set by testing multiple hyperparameter and Network sizes and comparing their
        # performance

        # Input dimension and first Hidden Layer
        model.add(Dense(hl1_dims, input_dim=input_layer_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Second Hidden Layer
        model.add(Dense(hl2_dims))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Third Hidden Layer
        model.add(Dense(hl3_dims))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # Output Layer
        model.add(Dense(output_layer_size))
        model.add(Activation('linear'))

        model.compile(optimizer=optimizer, loss=loss)  # Use Huber Loss Function for DQN based on TensorBoard analysis

        return model

    def _update_target_network_weights(self):
        self.dqn_target.set_weights(self.dqn_eval.get_weights())
