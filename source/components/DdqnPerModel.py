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


class DdqnPerModel(object):

    def __init__(self, name, alpha, gamma, input_layer_size, number_of_parameters, out_layer_size, memory_size=50000,
                 batch_size=64):

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        config.sections()

        enable_model_load = config['model_weights'].getboolean('enable_load_model_weights')
        self.enable_model_save = config['model_weights'].getboolean('enable_save_model_weights')
        self.tensorboard_visualization = config['tensorboard'].getboolean('enable_ddqnper')

        # Hyperparameters
        self.memory = MemoryBuffer(max_size=memory_size, number_of_parameters=input_layer_size, with_per=True)
        self.with_per = True
        self.gamma = gamma
        self.learning_rate = alpha
        self.batch_size = batch_size
        self.out_layer_size = out_layer_size
        self.replace_target_network_after = config['model_settings'].getint('ddqn_replace_network_interval')
        self.action_space = [i for i in range(out_layer_size)]
        self.priority_offset = 0.1  # used for priority, as we do not want to have priority 0 samples
        self.priority_scale = 0.7  # priority_scale, suggested by Paper

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
        # self.history = History()

        if self.tensorboard_visualization:
            comment = 'adam-huber-reward_per2'
            path = config['tensorboard']['file_path']
            tboard_name = '{}{}-cmt-{}_hl1_dims-{}_hl2_dims-{}_hl3_dims-{}-time-{}'.format(path, name, comment,
                                                                                           hl1_dims,
                                                                                           hl2_dims, hl3_dims,
                                                                                           int(time.time()))
            self.tensorboard = TensorBoard(tboard_name.format())

        self.keras_weights_filename = '{}.keras'.format(name)
        self.model_loaded = False
        if enable_model_load:
            self.load_model()
        else:
            print('Applying epsilon greedy strategy')

    def remember(self, previous_state, action, reward, new_state, done, street):
        self.memory.store_state(previous_state, action, reward, new_state, done)

    def replay(self):
        if not self.memory.mem_cntr > self.batch_size:
            return

        previous_states, actions, rewards, new_states, dones, importances, batch_indices = \
            self.memory.get_sample_batch(self.batch_size, self.priority_scale)

        q_val_prev = self.dqn_eval.predict(previous_states)
        q_target = q_val_prev  # needed to calculate differences which shall be zero
        q_val_new = self.dqn_eval.predict(new_states)  # Estimate new next action target given the state
        q_val_new_target = self.dqn_target.predict(new_states)

        next_actions_eval = np.argmax(q_val_new, axis=1)  # indices of max actions for q_val_new
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Update Q-Target
        q_target[batch_index, actions] = rewards + self.gamma * q_val_new_target[batch_index, next_actions_eval] * dones

        # Update active eval network with DDQN function using target network
        # PER: Apply Importance Weights to training. Square by 1 - episolon - the more the network knows, the more the
        # important samples should be trained on.
        sample_weight_importances = importances ** (1 - self.epsilon)
        if self.tensorboard_visualization:
            self.dqn_eval.fit(previous_states, q_target, verbose=0, epochs=1, callbacks=[self.tensorboard],
                              sample_weight=sample_weight_importances)
        else:
            self.dqn_eval.fit(previous_states, q_target, verbose=0, sample_weight=sample_weight_importances)

        # Update Priorities on batch_indices we just trained (this will change on whole buffer, therefore batch_indices
        # is needed, instead of sampled batch_index above.

        # Now compute temporal difference error for our predicted q_val and actual q_val after state transition
        q_next_action_target = q_val_new_target[batch_index, next_actions_eval]
        q_last_action_eval = q_target[batch_index, actions]
        td_errors = rewards + self.gamma * q_next_action_target - q_last_action_eval

        self.memory.set_priorities(batch_indices, td_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        if self.memory.mem_cntr % self.replace_target_network_after == 0:
            self._update_target_network_weights()

    def choose_action(self, state, street):
        """
        Choose the action to be taken for current state.
        Epsilon Greedy Policy is being applied here.
        If self.split_network is enabled, the street is being evaluated
        to decide which network to use.
        :param state: current state (feature vector)
        :param street: current street only used for split_network
        :return: action_index to perform action
        """

        if np.random.rand() <= self.epsilon:
            action = randint(0, self.out_layer_size - 1)
        else:
            size = state.shape
            state = np.reshape(state, [1, size[0]])
            q_values = self.dqn_eval.predict([state, np.ones(self.out_layer_size).reshape(1, self.out_layer_size)])
            action = np.argmax(q_values[0])

        return action

    def reset_state(self):
        self.dqn_eval.reset_states()

    def save_model(self):
        """
        If enable_model_save is set to true from config file, keras weights will be saved in /model
        """
        if not self.enable_model_save:
            return

        self.dqn_eval.save('./model/' + self.keras_weights_filename)
        print('Model weights have been saved to ./model/%f\n', self.keras_weights_filename)


    def load_model(self):
        """
        Load keras model weights under '/model' with keras_weights_filename
        """
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

    def _build_model(self, hl1_dims, hl2_dims, hl3_dims, input_layer_size, output_layer_size, optimizer, loss):
        """
        :param hl1_dims: dimensions for first hidden layer
        :param hl2_dims: dimensions for second hidden layer
        :param hl3_dims: dimensions for third hidden layer
        :param input_layer_size: dimensions for input layer
        :param output_layer_size: dimensions for output layer
        :param optimizer: optimizer that should be used
        :param loss: loss function that should be used
        :return: keras sequential with batch norm model
        """
        model = Sequential()

        # The Layer Size is being set by testing multiple hyperparameter and Network sizes and comparing their
        # performance based on RE5 Model Settings

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
