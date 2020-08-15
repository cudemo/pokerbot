import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Union, Tuple

import numpy as np
import tensorflow as tf
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

"""
Bot RE2 Deep Q Network v1 - 20.04.2020 deployed to AICrowd
Last stand before being integrated with other components as MVDqn.py
"""
class MVDeepQPlayer(BasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    def __init__(self):
        self.hole_cards = []
        self.number_of_montecarlo_simulations = 100
        self.number_of_independent_features = 6
        self.number_of_player_dependent_features = 3
        self.number_of_players = 0

        # Rules
        self.initial_stack = 0
        self.player_seat = 0

        # Counters
        self.blind_paid = 0
        self.my_bet = 0
        self.reward = 0
        self.cashgame_stack = 0

        # State
        self.uuid = ''
        self.folded = False
        self.last_action = False
        self.last_state = 0
        self.win_rate = 0
        self.times_won = 0
        self.times_folded = 0
        self.times_called = 0
        self.times_raised = 0
        self.round_count = 0
        self.folded_in_row_counter = 0
        self.won_in_row_counter = 0
        self.rounds_to_play = 0

        # Statistics
        self.opponent_statistics_aggressivity = {}
        self.opponent_statistics_tightness = {}

        # Hyperparameters
        self.memory = deque(maxlen=750)
        self.gamma = 0.93
        self.epsilon = 1.0  # enable epsilon = 1.0 only when changing model, else learned weights from .h5 are used.
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.input_layer_size = 9
        self.out_layer_size = 3
        self.batch_size = 64

        # Printouts - set all to False before uploading
        self.printouts = False
        self.episode_printout = True
        self.measure_declare_action_time = False

    def __str__(self):
        return "MV DeepQ"

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
        """
        Define what action the player should execute.
        :param valid_actions: List of dictionary containing valid actions the player can execute.
        :param hole_card: Cards in possession of the player encoded as a list of strings.
        :param round_state: Dictionary containing relevant information and history of the game.
        :return: action: str specifying action type. amount: int action argument.
        """

        if self.measure_declare_action_time:
            t1 = int(round(time.time() * 1000))

        # Update win_rate, as game is not finished
        self.win_rate = self._get_win_rate(round_state)

        state = self._get_state(round_state)

        if self.last_action:
            self._remember(self.last_state, self.last_action, self.reward, state, 0)

        possible_moves = []
        for dict in valid_actions:
            possible_moves.append(dict["action"])

        if np.random.rand() <= self.epsilon:
            action = random.choice(possible_moves)
        else:
            act_values = self.model.predict([state, np.ones(self.out_layer_size).reshape(1, self.out_layer_size)])
            action = valid_actions[np.argmax(act_values[0])]['action']

            if self.printouts:
                print('Predicted act value by NN: {}'.format(act_values))
                print('Cleaned up action of NN: {}'.format(action))

        action, amount = self._validate_action(action, valid_actions)
        action_index = possible_moves.index(action)

        # Remember decision and corresponding state for further remember function
        self.last_action = action_index
        self.last_state = state

        if self.printouts:
            print('Feeding state: {} into model for prediction'.format(state))
            print('Action: {}, Amount: {}'.format(action, amount))

        if self.measure_declare_action_time:
            t2 = int(round(time.time() * 1000))
            print('Decision taken in: {} seconds'.format((t2 - t1) / 1000))

        return action, amount


    def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
        """
        Called once the game started.
        :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
        """
        self.uuid = self.uuid
        self.initial_stack = game_info['rule']['initial_stack']
        self.small_blind_amount = game_info['rule']['small_blind_amount']
        self.big_blind_amount = 2 * self.small_blind_amount
        self._create_opponent_statistics_aggressivity(game_info['seats'])
        self._create_opponent_statistics_tightness(game_info['seats'])
        self.player_seat = game_info['player_num'] - 1
        self.number_of_players = len(game_info['seats'])
        self.rounds_to_play = game_info['rule']['max_round']

        self.input_layer_size = self.number_of_independent_features + self.number_of_players * self.number_of_player_dependent_features
        self.model = self._build_model()

        model = Path("./model/deepq.h5")
        if model.is_file():
            try:
                self.model.load_weights("./model/deepq.h5")
                self.epsilon = 0.1
            except ValueError:
                if self.printouts:
                    print("Model could not be loaded ... using epsilon greedy strategy")


    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict[str, Union[str, int]]]) -> None:
        """
        Called once a round starts.
        :param round_count: Round number, in Cash Game always 1.
        :param hole_card: Cards in possession of the player.
        :param seats: Players at the table.
        """
        self.hole_cards = hole_card
        self.reward = 0
        self.my_bet = 0
        self.folded = False
        self.done = 0
        self.round_count = round_count

        # Safety net
        if self.times_won / round_count < 0.01 and self.folded_in_row_counter > 8:
            self.epsilon = 0.4


    def receive_street_start_message(self, street: str, round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called at every stage (preflop, flop, turn, river, showdown).
        :param street: Game stage
        :param round_state: Dictionary containing the round state
        """
        pass


    def receive_game_update_message(self, action: Dict[str, Union[str, int]],
                                    round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called after every action made by any of the players.
        :param action: Dict containing the player uuid and the executed action
        :param round_state: Dictionary containing the round state
        """
        if action['player_uuid'] == self.uuid:
            return

        if action['action'] == 'raise':
            self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_raises'] = \
                self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_raises'] + 1

        self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_actions'] = \
            self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_actions'] + 1

        self.opponent_statistics_aggressivity[action['player_uuid']]['aggressivity'] = \
            self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_raises'] / \
            self.opponent_statistics_aggressivity[action['player_uuid']]['number_of_actions']

        if round_state['street'] == 'preflop':
            if action['action'] == 'fold':
                self.opponent_statistics_tightness[action['player_uuid']]['hands_folded'] = \
                    self.opponent_statistics_tightness[action['player_uuid']]['hands_folded'] + 1

            self.opponent_statistics_tightness[action['player_uuid']]['tightness'] = \
                self.opponent_statistics_tightness[action['player_uuid']]['hands_folded'] / round_state['round_count']


    def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                                     hand_info: [List[Dict[str, Union[str, Dict]]]],
                                     round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Called at the end of the round.
        :param winners: List of the round winners containing the stack and player information.
        :param hand_info: List containing a Dict for every player at the table describing the players hand this round.
        :param round_state: Dictionary containing the round state
        """
        done = 1
        won = 0
        game_stack = self.initial_stack - self.my_bet
        self.cashgame_stack -= self.my_bet

        if self.folded:
            self.reward = 2 * ((1.001 + game_stack) * (1.001 - self.win_rate)**2)
        else:
            self.reward = 2 * (-(1.001 - game_stack) * (1.001 - self.win_rate)**4) # higher the negative reward if not won
            for winner in winners:
                if winner['uuid'] == self.uuid:
                    won = 1
                    self.reward = winner['stack'] - self.my_bet
                    self.cashgame_stack = winner['cashgame_stack'] + self.reward
                    self.times_won += 1
                    print("WON")
                    break

        # Update counters
        if won:
            self.won_in_row_counter += 1
            self.folded_in_row_counter = 0
        elif not self.folded:
            self.folded_in_row_counter = 0
            self.won_in_row_counter = 0
        else:
            self.folded_in_row_counter += 1

        next_state = self._get_state(round_state)
        self._remember(self.last_state, self.last_action, self.reward, next_state, done)

        replay_after_50_games = True if round_state['round_count'] % 50 == 0 else False

        if len(self.memory) > self.batch_size and replay_after_50_games:
            self._replay(self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.episode_printout:
            print("Episode: {}, reward: {:6.2f}, folded: {}, cashgame stack: {:6.2f},  won: {}".format(round_state['round_count'], self.reward, self.folded, self.cashgame_stack, won))

        if self.rounds_to_play == self.round_count:
            self.model.save("./model/deepq.h5")
            print('\nGame Statistics:')
            print('Times folded: {}'.format(self.times_folded))
            print('Times called: {}'.format(self.times_called))
            print('Times raised: {}'.format(self.times_raised))
            print('Percentage of games won: {}%'.format(self.times_won * 100 / self.rounds_to_play))
            print('Model weights have been saved to /model/deepq.h5 \n')

    def _build_model(self):
        model = Sequential()

        model.add(Dense(self.input_layer_size, input_dim=self.input_layer_size))
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(self.out_layer_size))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state) # Estimate new next action target given the state
            target_f[0][action] = target  # Map target from current state to future state
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def _get_win_rate(self, round_state):
        # Calculate profit probability using Monte Carlo simulation
        nb_player = len(round_state['seats'])
        community_card = round_state['community_card']

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.number_of_montecarlo_simulations,
            nb_player=nb_player,
            hole_card=gen_cards(self.hole_cards),
            community_card=gen_cards(community_card)
        )
        return win_rate

    def _validate_action(self, action, valid_actions):
        amount = 0

        if action == 'fold':
            self.folded = True
            self.times_folded += 1

        if action == 'raise':
            # Choose a random amount within the allowed range to raise
            action = valid_actions[2]['action']  # fetch RAISE action info
            amount_limits = valid_actions[2]['amount']
            # amount can be -1 if it is no longer possible to raise, must call instaed
            if amount == -1:
                amount = valid_actions[1]['amount']
            else:
                # Calculate raise based on probability of winning
                range = max(amount_limits["min"], amount_limits["max"]) - amount_limits["min"]
                amount = amount_limits["min"] + self.win_rate * range
                if self.printouts:
                    print('Amount set to: ', amount)

            self.times_raised += 1
                # amount = random.randint(amount_limits["min"], max(amount_limits["min"], amount_limits["max"]))

        if action == 'call':
            action = valid_actions[1]['action']
            amount = valid_actions[1]['amount']
            self.times_called += 1

        self.my_bet = amount
        return action, amount

    def _create_opponent_statistics_aggressivity(self, seats):
        for seat in seats:
            if seat['uuid'] == self.uuid:
                continue

            self.opponent_statistics_aggressivity[seat['uuid']] = {
                'number_of_raises': 0,
                'number_of_actions': 0,
                'aggressivity': 0
            }

    def _create_opponent_statistics_tightness(self, seats):
        for seat in seats:
            if seat['uuid'] == self.uuid:
                continue

            self.opponent_statistics_tightness[seat['uuid']] = {
                'hands_folded': 0,
                'tightness': 0
            }


    def _get_position_one_hot(self, round_state):
        early, middle, late = 0, 0, 0
        active_players = self._get_active_players(round_state['seats'], True)
        index_own_player = self._get_index_of_player(round_state['seats'], self.uuid)

        distance = 0
        cursor = round_state['small_blind_pos']
        while cursor % len(active_players) != index_own_player:
            cursor += 1
            distance += 1
        relative_distance_from_smallblind = distance / len(active_players)

        if len(active_players) == 2:
            # Special case heads-up
            if relative_distance_from_smallblind == 0:
                early = 1
            else:
                late = 1
        else:
            # Every other number of participants
            if relative_distance_from_smallblind < 0.3:
                early = 1
            elif relative_distance_from_smallblind < 0.6:
                middle = 1
            else:
                late = 1

        return [early, middle, late]

    def _get_street_one_hot(self, round_state):
        street_index = {
            'preflop': [1, 0, 0, 0, 0],
            'flop': [0, 1, 0, 0, 0],
            'turn': [0, 0, 1, 0, 0],
            'river': [0, 0, 0, 1, 0],
            'showdown': [0, 0, 0, 0, 1]
        }

        return street_index[round_state['street']]

    def _get_active_players(self, seats, include_own):
        active_players = []

        for seat in seats:
            if seat['uuid'] == self.uuid and not include_own:
                continue
            if seat['state'] == 'folded':
                continue

            active_players.append(seat['uuid'])

        return active_players

    def _get_state(self, round_state):
        # Get Features
        street_one_hot = self._get_street_one_hot(round_state)
        position_one_hot = self._get_position_one_hot(round_state)
        statistics = self._get_player_statistics(round_state['seats'])

        features = [self.win_rate] + street_one_hot + position_one_hot + statistics
        state = np.array(features)
        state = np.reshape(state, [1, self.input_layer_size])
        return tf.keras.backend.cast_to_floatx(state)

    def _get_player_statistics(self, seats):
        """
        Returns a list, containing the following features for each opponent,
        starting with the player to our left, going clockwise:
        - is player active
        - aggressivity of player
        - tightness of player
        """
        statistics = []
        number_of_opponents = self.number_of_players - 1
        number_of_statistics_features = 3
        index_of_own_player = self._get_index_of_player(seats, self.uuid)

        current_index = index_of_own_player
        while len(statistics) < number_of_opponents * number_of_statistics_features:
            current_index += 1
            if seats[current_index % self.number_of_players]['uuid'] == self.uuid:
                continue

            statistics.append(0 if seats[current_index % self.number_of_players]['state'] == 'folded' else 1)
            statistics.append(self.opponent_statistics_aggressivity[seats[current_index % self.number_of_players]['uuid']]['aggressivity'])
            statistics.append(self.opponent_statistics_tightness[seats[current_index % self.number_of_players]['uuid']]['tightness'])

        return statistics

    def _get_index_of_player(self, seats, uuid):
        index_of_own_player = 0
        for seat in seats:
            if seat['uuid'] == uuid:
                return index_of_own_player
            index_of_own_player += 1
