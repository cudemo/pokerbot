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


class DdqnPerSplitModel(object):

    def __init__(self, name, alpha, gamma, input_layer_size, input_layer_preflop_size, out_layer_size,
                 split_network=True, with_per=True, memory_size=50000, batch_size=32):

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        config.sections()

        enable_model_load = config['model_weights'].getboolean('enable_load_model_weights')
        self.enable_model_save = config['model_weights'].getboolean('enable_save_model_weights')
        self.tensorboard_visualization = config['tensorboard'].getboolean('enable_ddqnper')
        self.reset_epsilon_greedy_if_model_loaded_to = config['model_settings'].getfloat(
            'reset_epsilon_greedy_if_model_loaded_to')

        # Hyperparameters
        self.memory = MemoryBuffer(max_size=memory_size, number_of_parameters=input_layer_size, with_per=with_per)
        if split_network:
            self.memory_preflop = MemoryBuffer(max_size=memory_size, number_of_parameters=input_layer_preflop_size,
                                               with_per=with_per)
        self.gamma = gamma
        self.learning_rate = alpha
        self.batch_size = batch_size
        self.out_layer_size = out_layer_size
        self.replace_target_network_after = config['model_settings'].getint('ddqn_replace_network_interval')
        self.priority_offset = 0.1  # used for priority, as we do not want to have priority 0 samples
        self.priority_scale = 0.7  # priority_scale, suggested by Paper
        self.split_network = split_network
        self.with_per = with_per

        loss = Huber()
        optimizer = Adam(learning_rate=alpha)

        # Epsilon Greedy Strategy
        self.epsilon = 1.0  # enable epsilon = 1.0 only when changing model, else learned weights from .h5 are used.
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.005

        # Keras Models
        if split_network:
            hl1_dims = 512
            hl2_dims = 256
            hl3_dims = 128

            hl1_preflop_dims = 128
            hl2_preflop_dims = 64
            hl3_preflop_dims = 64

            self.dqn_eval_preflop = self._build_model(hl1_preflop_dims, hl2_preflop_dims, hl3_preflop_dims,
                                                      input_layer_preflop_size, out_layer_size, optimizer, loss)
            self.dqn_target_preflop = self._build_model(hl1_preflop_dims, hl2_preflop_dims, hl3_preflop_dims,
                                                        input_layer_preflop_size, out_layer_size, optimizer, loss)

            self.keras_weights_preflop_filename = '{}_preflop.keras'.format(name)
        else:
            hl1_dims = 128
            hl2_dims = 64
            hl3_dims = 64

        self.dqn_eval = self._build_model(hl1_dims, hl2_dims, hl3_dims, input_layer_size, out_layer_size, optimizer,
                                          loss)
        self.dqn_target = self._build_model(hl1_dims, hl2_dims, hl3_dims, input_layer_size, out_layer_size, optimizer,
                                            loss)

        self.keras_weights_filename = '{}.keras'.format(name)

        if self.tensorboard_visualization:
            comment = 'adam-huber'
            comment_preflop = 'adam-huber-preflop'
            path = config['tensorboard']['file_path']
            tboard_name = '{}{}-cmt-{}_hl1_dims-{}_hl2_dims-{}_hl3_dims-{}-time-{}'.format(path, name, comment,
                                                                                           hl1_dims,
                                                                                           hl2_dims, hl3_dims,
                                                                                           int(time.time()))

            if split_network:
                tboard_name_preflop = '{}{}-cmt-{}_hl1_dims-{}_hl2_dims-{}_hl3_dims-{}-time-{}'.format(path, name,
                                                                                                       comment_preflop,
                                                                                                       hl1_preflop_dims,
                                                                                                       hl2_preflop_dims,
                                                                                                       hl3_preflop_dims,
                                                                                                       int(time.time()))
                self.tensorboard_preflop = TensorBoard(tboard_name_preflop.format())

            self.tensorboard = TensorBoard(tboard_name.format())


        self.model_loaded = False
        if enable_model_load:
            self.load_model()
        else:
            print('Applying epsilon greedy strategy')

    def remember(self, previous_state, action, reward, new_state, done, street):
        """
        :param previous_state: previous feature state
        :param action: action taken in that state
        :param reward: reward gained by that action in that state
        :param new_state: new_state after one step
        :param done: terminal value for the round
        :param street: street, used for split network
        """
        if street == 'preflop' and self.split_network:
            self.memory_preflop.store_state(previous_state, action, reward, new_state, done)
        else:
            self.memory.store_state(previous_state, action, reward, new_state, done)

    def replay(self):
        """
        Replay sample to apply model fitting.
        This replay function checks if split network is enabled and correspondingly applies replay to all networks.
        """

        if self.with_per:
            self.replay_network_per()
            if self.split_network:
                self.replay_preflop_network_per()
        else:
            self.replay_network()
            if self.split_network:
                self.replay_preflop_network()

    def replay_network_per(self):
        """
        Replay sample for network to apply model fitting while using Priotizied Experience Replay.
        Get MemoryBuffer sample and use Q-Learning Function applying TD-Error for Prioritized Replay
        """

        if not self.memory.mem_cntr > self.batch_size:
            return

        if self.with_per:
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

    def replay_preflop_network_per(self):
        """
        Replay sample for preflop network (if self.split_network is enabled) to apply model fitting while using
        Priotizied Experience Replay.
        Get MemoryBuffer sample and use Q-Learning Function applying TD-Error for Prioritized Replay
        """

        if not self.memory_preflop.mem_cntr > self.batch_size or not self.split_network:
            return

        previous_states, actions, rewards, new_states, dones, importances, batch_indices = \
            self.memory_preflop.get_sample_batch(self.batch_size, self.priority_scale)

        q_val_prev = self.dqn_eval_preflop.predict(previous_states)
        q_target = q_val_prev  # needed to calculate differences which shall be zero
        q_val_new = self.dqn_eval_preflop.predict(new_states)  # Estimate new next action target given the state
        q_val_new_target = self.dqn_target_preflop.predict(new_states)
        next_actions_eval = np.argmax(q_val_new, axis=1)  # indices of max actions for q_val_new
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Update Q-Target
        q_target[batch_index, actions] = rewards + self.gamma * q_val_new_target[batch_index, next_actions_eval] * dones

        # Update active eval network with DDQN function using target network
        # PER: Apply Importance Weights to training. Square by 1 - episolon - the more the network knows, the more the
        # important samples should be trained on.
        sample_weight_importances = importances ** (1 - self.epsilon)

        if self.tensorboard_visualization:
            self.dqn_eval_preflop.fit(previous_states, q_target, verbose=0, epochs=1,
                                      callbacks=[self.tensorboard_preflop], sample_weight=sample_weight_importances)
        else:
            self.dqn_eval_preflop.fit(previous_states, q_target, verbose=0, sample_weight=sample_weight_importances)

        # Update Priorities on batch_indices we just trained (this will change on whole buffer, therefore batch_indices
        # is needed, instead of sampled batch_index above.

        # Now compute temporal difference error for our predicted q_val and actual q_val after state transition
        q_next_action_target = q_val_new_target[batch_index, next_actions_eval]
        q_last_action_eval = q_target[batch_index, actions]
        td_errors = rewards + self.gamma * q_next_action_target - q_last_action_eval

        self.memory_preflop.set_priorities(batch_indices, td_errors)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        if self.memory_preflop.mem_cntr % self.replace_target_network_after == 0:
            self._update_target_network_weights()

    def replay_network(self):
        """
        Replay sample for network to apply model fitting.
        Get MemoryBuffer sample and use Q-Learning Function.
        """

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

    def replay_preflop_network(self):
        """
        Replay sample for preflop network to apply model fitting.
        Get MemoryBuffer sample and use Q-Learning Function.
        """

        if not self.memory_preflop.mem_cntr > self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory_preflop.get_sample_batch(self.batch_size)

        q_next = self.dqn_target_preflop.predict(new_states)
        q_eval = self.dqn_eval_preflop.predict(new_states)  # Estimate new next action target given the state
        q_pred = self.dqn_eval_preflop.predict(states)
        q_target = q_pred  # needed to calculate differences which shall be zero

        max_actions = np.argmax(q_eval, axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = \
            rewards + self.gamma * q_next[batch_index, max_actions.astype(
                int)] * dones  # done has been inverted, see MemoryBuffer terminal_memory

        # Update active eval network with DDQN function using target network
        if self.tensorboard_visualization:
            self.dqn_eval_preflop.fit(states, q_target, verbose=0, epochs=1, callbacks=[self.tensorboard_preflop])
        else:
            self.dqn_eval_preflop.fit(states, q_target, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

        if self.memory_preflop.mem_cntr % self.replace_target_network_after == 0:
            self._update_preflop_target_network_weights()

    def choose_action(self, state, street):
        """
        Choose the action to be taken for current state.
        Epsilon Greedy Policy is being applied here.
        If self.split_network is enabled, the street is being evaluated
        to decide which network to use.
        :param state: current state (feature vector)
        :param street: current street for split_network
        :return: action_index to perform action
        """

        if np.random.rand() <= self.epsilon:
            action = randint(0, self.out_layer_size - 1)
        else:
            size = state.shape
            state = np.reshape(state, [1, size[0]])

            if street == 'preflop' and self.split_network:
                q_values = self.dqn_eval_preflop.predict([state, np.ones(self.out_layer_size).reshape(1, self.out_layer_size)])
            else:
                q_values = self.dqn_eval.predict([state, np.ones(self.out_layer_size).reshape(1, self.out_layer_size)])
            action = np.argmax(q_values[0])

        return action

    def reset_state(self):
        self.dqn_eval.reset_states()

    def save_model(self):
        """
        If self.enable_model_save (from agent_config.ini) is enabled, the model weights will be saved to location
        provided.
        Also checks if split_network is enabled and saves corresponding weights.
        """
        if not self.enable_model_save:
            return

        self.dqn_eval.save('./model/' + self.keras_weights_filename)
        print('Model weights have been saved to ./model/' + self.keras_weights_filename)
        if self.split_network:
            self.dqn_eval_preflop.save('./model/' + self.keras_weights_preflop_filename)
            print('Model preflop weights have been saved to ./model/' + self.keras_weights_preflop_filename)

    def load_model(self):
        """
        Load keras model weights under '/model' with keras_weights_filename
        """
        if Path('./model/' + self.keras_weights_filename).is_file():
            try:
                self.dqn_eval.load_weights('./model/' + self.keras_weights_filename)
                self.epsilon = self.reset_epsilon_greedy_if_model_loaded_to
                self.model_loaded = True
                print('Pretrained model loaded, epsilon greedy set to {}'
                      .format(self.reset_epsilon_greedy_if_model_loaded_to))
            except ValueError:
                print("Model could not be loaded ... using epsilon greedy strategy")
                self.epsilon = 1.0

            if self.split_network and Path('./model/' + self.keras_weights_preflop_filename).is_file():
                try:
                    self.dqn_eval_preflop.load_weights('./model/' + self.keras_weights_preflop_filename)
                    print('Pretrained preflop model loaded, epsilon greedy set to {}'
                          .format(self.reset_epsilon_greedy_if_model_loaded_to))
                except ValueError:
                    print("Model could not be loaded ... using epsilon greedy strategy")
                    self.epsilon = self.reset_epsilon_greedy_if_model_loaded_to
        else:
            print('No model to load, applying epsilon greedy strategy')

        self._update_target_network_weights()
        self._update_preflop_target_network_weights()

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

        model.compile(optimizer=optimizer, loss=loss)

        return model

    def _update_target_network_weights(self):
        self.dqn_target.set_weights(self.dqn_eval.get_weights())

    def _update_preflop_target_network_weights(self):
        if self.split_network:
            self.dqn_target_preflop.set_weights(self.dqn_eval_preflop.get_weights())
