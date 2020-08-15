import configparser
import time
from pathlib import Path

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import BatchNormalization, Flatten
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.optimizer_v2.adam import Adam


class DdqnKerasModel(object):
    """
    Bot Double Deep Q Network from Keras.agents.dqn, implemented for reference and local training
    """

    def __init__(self, name, alpha, gamma, input_layer_size, number_of_parameters, out_layer_size, memory_size=50000,
                 batch_size=32):

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        config.sections()

        enable_model_load = config['model_weights'].getboolean('enable_load_model_weights')
        self.enable_model_save = config['model_weights'].getboolean('enable_save_model_weights')
        self.tensorboard_visualization = config['tensorboard'].getboolean('enable_ddqn_keras')

        # Hyperparameters
        self.with_per = True
        self.gamma = gamma
        self.learning_rate = alpha
        self.out_layer_size = out_layer_size
        self.action_space = [i for i in range(out_layer_size)]
        self.offset = 0.1  # used for priority, as we do not want to have priority 0 samples
        replace_target_network_after = config['model_settings'].getint('ddqn_replace_network_interval')

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

        self.keras_weights_filename = '{}.keras'.format(name)

        network_model = self._build_model(hl1_dims, hl2_dims, hl3_dims, input_layer_size, out_layer_size, optimizer, loss)
        # self.history = History()
        self.agent = 0

        self.memory = SequentialMemory(memory_size, window_length=1)
        self.model_loaded = False

        self.load_model(network_model, out_layer_size, replace_target_network_after, optimizer, enable_model_load)

        if self.tensorboard_visualization:
            comment = 'adam-huber-reward_per5'
            path = config['tensorboard']['file_path']
            tboard_name = '{}{}-cmt-{}_hl1_dims-{}_hl2_dims-{}_hl3_dims-{}-time-{}'.format(path, name, comment,
                                                                                           hl1_dims,
                                                                                           hl2_dims, hl3_dims,
                                                                                           int(time.time()))
            self.tensorboard = TensorBoard(tboard_name.format())


    def choose_action(self, state, street):
        action = self.agent.forward(state[0:])
        return action

    def reset_state(self):
        self.agent.reset_states()

    def remember(self, previous_state, action, reward, new_state, done, street):
        self.agent.backward(reward, done)

    def replay(self):
        if self.memory.nb_entries > 100:
            self.agent.step = 100

    def save_model(self):
        if not self.enable_model_save:
            return

        self.agent.save_weights('./model/' + self.keras_weights_filename, overwrite=True)
        print('Model weights have been saved to ./model/%f\n', self.keras_weights_filename)

        # if self.epsilon <= self.epsilon_min: #Target network update if epsilon already near zero
        #   self._update_target_network_weights()

    def load_model(self, network_model, out_layer_size, replace_target_network_after, optimizer, enable_model_load):
        model = Path('./model/' + self.keras_weights_filename)
        if enable_model_load and model.is_file():
            try:
                policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05,
                                              value_min=self.epsilon_min,
                                              value_test=.05, nb_steps=3000)
                self.agent = DQNAgent(model=network_model, nb_actions=out_layer_size, memory=self.memory, nb_steps_warmup=1,
                                      target_model_update=replace_target_network_after, policy=policy,
                                      enable_double_dqn=True)
                self.agent.compile(optimizer=optimizer)
                self.agent.load_weights('./model/' + self.keras_weights_filename)
                self.model_loaded = True
                print('Pretrained model loaded, epsilon greedy set to 0.05')
            except ValueError:
                print("Model could not be loaded ... using epsilon greedy strategy")
        else:
            print('Applying epsilon greedy strategy')
            policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=self.epsilon,
                                          value_min=self.epsilon_min,
                                          value_test=.05, nb_steps=3000)

            self.agent = DQNAgent(model=network_model, nb_actions=out_layer_size, memory=self.memory,
                                  nb_steps_warmup=1, target_model_update=replace_target_network_after, policy=policy,
                                  enable_double_dqn=True)

            self.agent.compile(optimizer=optimizer)

    def _build_model(self, hl1_dims, hl2_dims, hl3_dims, input_layer_size, output_layer_size, optimizer, loss):

        input_v = Input(shape=(1, input_layer_size))

        branch_v = Flatten()(input_v)
        branch_v = Dense(hl1_dims, activation='relu')(branch_v)
        branch_v = BatchNormalization()(branch_v)
        branch_v = Dense(hl2_dims, activation='relu')(branch_v)
        branch_v = BatchNormalization()(branch_v)
        out_v = Dense(hl3_dims, activation='relu')(branch_v)
        out_v = BatchNormalization()(out_v)
        out_v = Dense(output_layer_size, activation='linear')(out_v)
        model = Model(inputs=input_v, outputs=out_v)

        #model = Model(inputs=m_v.inputs, outputs=out_v)
        # print(model.summary())

        model.compile(optimizer=optimizer, loss=loss)  # Use Huber Loss Function for DQN based on TensorBoard analysis

        return model

    """
    def _build_model(self, hl1_dims, hl2_dims, hl3_dims, input_layer_size, output_layer_size, optimizer, loss):

        # The Layer Size is being set by testing multiple hyperparameter and Network sizes and comparing their
        # performance
        # Shape: a shape (30,4,10) means an array or tensor with 3 dimensions, containing 30 elements in the first
        # dimension, 4 in the second and 10 in the third, totaling 30*4*10 = 1200 elements or numbers.
        # Our feature shape: (1, 1, 1, 4, 3, 4, 10, 5, 5) - 9 dimensions

        # Setup Input Layers for feature state based on sizes of each feature
        input_v = Input(shape=(1, 56))
        input_3d = Input(shape=(3,))
        input_4d = Input(shape=(4,))
        input_5d = Input(shape=(5,))
        input_10d = Input(shape=(10,))

        branch_v = Flatten()(input_v)
        branch_v = Dense(128, activation='relu')(branch_v)
        branch_v = Dense(64, activation='relu')(branch_v)
        out_v = Dense(16, activation='relu')(branch_v)
        out_v = Dense(output_layer_size, activation='linear')(out_v)
        m_v = Model(inputs=input_v, outputs=out_v)

        # Opponents
        branch_3d = Dense(32, activation='relu')(input_3d)
        branch_3d = Dense(32, activation='relu')(branch_3d)
        m_3d = Model(inputs=input_3d, outputs=branch_3d)

        # Street, Position
        branch_4d = Dense(32, activation='relu')(input_4d)
        branch_4d = Dense(32, activation='relu')(branch_4d)
        m_4d = Model(inputs=input_4d, outputs=branch_4d)

        # Aggressivity, Tightness, pot_odds
        branch_5d = Dense(32, activation='relu')(input_5d)
        branch_5d = Dense(32, activation='relu')(branch_5d)
        m_5d = Model(inputs=input_5d, outputs=branch_5d)

        # Street, Position, pot_size, bet_size
        branch_10d = Dense(128, activation='relu')(input_10d)
        branch_10d = Dense(64, activation='relu')(branch_10d)
        branch_10d = Dense(32, activation='relu')(branch_10d)
        m_10d = Model(inputs=input_10d, outputs=branch_10d)

        # Combine all output of branches
        combined = Concatenate(axis=1)([m_3d.output, m_4d.output, m_5d.output, m_10d.output])

        # Apply FC Layer
        out = Dense(16, activation='relu')(combined)
        out = Dense(output_layer_size, activation='linear')(out)

        # Model accepts inputs of all branches and output action space based on output_layer_size
        #model = Model(inputs=[m_3d.input, m_4d.input, m_5d.input, m_10d.input], outputs=out)
        model = Model(inputs=m_v.inputs, outputs=out_v)
        #print(model.summary())

        model.compile(optimizer=optimizer, loss=loss)  # Use Huber Loss Function for DQN based on TensorBoard analysis

        return model
    """
