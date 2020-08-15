from typing import Dict, Union, List

import numpy as np
from treys import Evaluator, Card

from components.HoldemCalc import Card as HoldemCalcCard
from components.HoldemCalc import HoldemCalc
from components.Players import Players


class Features:

    # Source for hand ranking dictionaries: https://wizardofodds.com/games/texas-hold-em/6-player-game/
    # Pocket pairs
    hand_values_pairs = {
        '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 13, '8': 15, '9': 18, '10': 21, '11': 25, '12': 29, '13': 34, '14': 40
    }

    # Suited hole cards
    hand_values_suited = {
        '2': {'3': 4, '4': 5, '5': 5, '6': 5, '7': 4, '8': 5, '9': 5, '10': 6, '11': 7, '12': 8, '13': 9, '14': 12},
        '3': {'4': 6, '5': 7, '6': 6, '7': 5, '8': 5, '9': 6, '10': 7, '11': 7, '12': 8, '13': 10, '14': 12},
        '4': {'5': 8, '6': 7, '7': 7, '8': 6, '9': 6, '10': 7, '11': 8, '12': 9, '13': 10, '14': 13},
        '5': {'6': 8, '7': 8, '8': 8, '9': 7, '10': 7, '11': 8, '12': 9, '13': 11, '14': 13},
        '6': {'7': 9, '8': 9, '9': 9, '10': 9, '11': 9, '12': 10, '13': 11, '14': 13},
        '7': {'8': 10, '9': 10, '10': 10, '11': 10, '12': 10, '13': 12, '14': 14},
        '8': {'9': 12, '10': 12, '11': 12, '12': 12, '13': 13, '14': 14},
        '9': {'10': 14, '11': 14, '12': 14, '13': 14, '14': 15},
        '10': {'11': 16, '12': 16, '13': 17, '14': 18},
        '11': {'12': 17, '13': 18, '14': 19},
        '12': {'13': 19, '14': 20},
        '13': {'14': 22},
    }

    # Not suited hole cards
    hand_values_offsuit = {
        '2': {'3': 0, '4': 1, '5': 1, '6': 1, '7': 0, '8': 1, '9': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 8},
        '3': {'4': 2, '5': 3, '6': 2, '7': 1, '8': 1, '9': 1, '10': 2, '11': 3, '12': 4, '13': 6, '14': 8},
        '4': {'5': 4, '6': 3, '7': 3, '8': 2, '9': 2, '10': 3, '11': 4, '12': 5, '13': 6, '14': 9},
        '5': {'6': 5, '7': 4, '8': 4, '9': 3, '10': 3, '11': 4, '12': 5, '13': 7, '14': 9},
        '6': {'7': 5, '8': 5, '9': 5, '10': 5, '11': 5, '12': 6, '13': 7, '14': 9},
        '7': {'8': 7, '9': 6, '10': 6, '11': 6, '12': 6, '13': 8, '14': 10},
        '8': {'9': 8, '10': 8, '11': 8, '12': 8, '13': 9, '14': 10},
        '9': {'10': 10, '11': 10, '12': 10, '13': 11, '14': 11},
        '10': {'11': 13, '12': 13, '13': 13, '14': 14},
        '11': {'12': 14, '13': 15, '14': 15},
        '12': {'13': 16, '14': 17},
        '13': {'14': 19},
    }

    # Lookup dictionary for different one-hot encoded features in ten percent steps
    ten_percent_one_hot = {
        '0%-10%': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        '10%-20%': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        '20%-30%': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        '30%-40%': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        '40%-50%': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        '50%-60%': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        '60%-70%': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
        '70%-80%': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
        '80%-90%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
        '90%-100%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    }

    # One-hot encoding for feature street
    street_one_hot = {
        'preflop': np.array([1, 0, 0, 0]),
        'flop': np.array([0, 1, 0, 0]),
        'turn': np.array([0, 0, 1, 0]),
        'river': np.array([0, 0, 0, 1])
    }

    # One-hot encoding for number of players
    number_of_opponents_one_hot = {
        2: np.array([1, 0, 0, 0]),
        3: np.array([0, 1, 0, 0]),
        4: np.array([0, 0, 1, 0]),
        5: np.array([0, 0, 0, 1])
    }

    def __init__(self):
        self.opponent_statistics_aggressivity = {}
        self.opponent_statistics_tightness = {}

    def get_hand_rank(self, hole_card, round_state, debug_printouts):
        """
        :param hole_card: Hole cards of the own player
        :param round_state: Current round state to check street und community cards
        :param debug_printouts: Parameter for debugging purpose only. Allows printing the calculated hand rank
        :return: Relative hand rank amongst all possible hands represented als float between 0 and 1
        """
        if round_state['street'] == 'preflop':
            return self._get_hole_card_rank(hole_card)
        else:
            return self._get_cards_rank(hole_card, round_state, debug_printouts)

    def get_hand_rank_one_hot(self, hole_card, round_state, debug_printouts):
        """
        :param hole_card: Hole cards of the own player
        :param round_state: Current round state to check street und community cards
        :param debug_printouts: Parameter for debugging purpose only. Allows printing the calculated hand rank
        :return: One-hot encoded hand rank in five percent steps, represented by a binary array of size 20
        """
        hand_rank = self.get_hand_rank(hole_card, round_state, debug_printouts)
        return self._get_five_percent_one_hot(hand_rank)

    def get_hand_probabilities_histogram(self, hole_cards, board):
        """
        :param hole_cards: Hole cards of the own player
        :param board: Community cards represented by a list of strings
        :return: Array of ten floats, representing the chances to get a specific hand strength (High card, Pair, etc.)
        """
        hole_cards = (HoldemCalcCard(hole_cards[0]), HoldemCalcCard(hole_cards[1]))
        board_calc = []
        for board_card in board:
            board_calc.append(HoldemCalcCard(board_card))

        holdem_calc = HoldemCalc()
        return holdem_calc.get_hand_probabilities(hole_cards, board_calc)

    def get_hand_probabilities_histogram_one_hot(self, hole_cards, board):
        """
        :param hole_cards: Hole cards of the own player
        :param board: Community cards represented by a list of strings
        :return: One-hot encoded chance to get a specific hand strength (High card, Pair, Two Pairs, etc.).
        The chance for each hand strength is split in ten percent steps, represented by a binary array
        """
        probabilities = self.get_hand_probabilities_histogram(hole_cards, board)

        one_hot_probabilities = []
        for probability in probabilities:
            one_hot_probabilities.extend(self._get_ten_percent_one_hot(probability))

        return np.array(one_hot_probabilities)

    def get_street_one_hot(self, street):
        """
        :param street: Current street as string
        :return: One-hot encoded binary array of the current street
        """
        return self.street_one_hot[street]

    def get_position_one_hot(self, round_state, own_uuid):
        """
        :param round_state: Current round state containing the seats and small blind position
        :param own_uuid: Uuid of the own player
        :return: One-hot encoded relative distance from the small blind.
        Represented by a binary encoding distinguishing between early, middle and late position
        """
        early, middle, late = 0, 0, 0

        players = Players()
        active_players = players.get_active_players(round_state['seats'], True, own_uuid)
        index_own_player = players.get_index_of_player(round_state['seats'], own_uuid, True)
        distance = 0
        cursor = round_state['small_blind_pos']
        while cursor % len(active_players) != index_own_player:
            cursor += 1
            distance += 1

            if distance > 2 * len(round_state['seats']):
                str_small_blind_pos = round_state['small_blind_pos']
                str_index_own_player = index_own_player
                str_seats = round_state['seats']
                raise IndexError('_get_position_one_hot used to many iterations.\n'
                                 'small_blind_pos: ' + str(str_small_blind_pos) + '\n'
                                 'index_own_player: ' + str(str_index_own_player) + '\n'
                                 'self.uuid: ' + own_uuid + '\n'
                                 'seats:' + str(str_seats))

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

        return np.array([early, middle, late])

    def get_number_of_opponents_one_hot(self, seats, own_uuid):
        """
        :param seats: All seats taking place in the poker game
        :param own_uuid: Uuid of the own player
        :return: One-hot encoded number of opponents, either 2, 3, 4 or 5
        """
        players = Players()
        number_of_opponents = len(players.get_active_players(seats, False, own_uuid))

        if number_of_opponents >= 5:
            return self.number_of_opponents_one_hot[5]
        elif number_of_opponents > 3:
            return self.number_of_opponents_one_hot[4]
        elif number_of_opponents > 2:
            return self.number_of_opponents_one_hot[3]
        else:
            return self.number_of_opponents_one_hot[2]

    def get_pot_size(self, pot, number_of_seats, initial_stack):
        """
        :param pot: Number of chips in the pot
        :param number_of_seats: Total players in the game, not depending on their current status
        :param initial_stack: Number of chips handed to each player at the start of a round
        :return: Float representing the size of the current pot in comparision to the maximal amount as number between
        0 and 1
        """
        return pot / (number_of_seats * initial_stack)

    def get_pot_size_one_hot(self, pot, number_of_seats, initial_stack):
        """
        :param pot: Number of chips in the pot
        :param number_of_seats: Total players in the game, not depending on their current status
        :param initial_stack: Number of chips handed to each player at the start of a round
        :return: One-hot encoded pot size in ten percent steps between 0 and maximal possible pot size (1200)
        """
        pot_size = self.get_pot_size(pot, number_of_seats, initial_stack)
        return self._get_ten_percent_one_hot(pot_size)

    def get_pot_odds(self, amount_to_play, pot):
        """
        :param amount_to_play: Required number of chips to stay in current round
        :param pot: Number of chips in the pot
        :return: Relative amount of chips to stay in current round compared to pot
        """
        return amount_to_play / pot

    def get_pot_odds_one_hot(self, amount_to_play, pot):
        """
        :param amount_to_play: Required number of chips to stay in current round
        :param pot: Number of chips in the pot
        :return: One-hot encoded relative amount of chips to stay in current round compared to pot.
        Possible values are free calls, up to 25%, up to 50%, up to 75% and above
        """
        pot_odds_one_hot = {
            'free': np.array([1, 0, 0, 0, 0]),
            'cheap': np.array([0, 1, 0, 0, 0]),
            'moderate': np.array([0, 0, 1, 0, 0]),
            'expensive': np.array([0, 0, 0, 1, 0]),
            'very expensive': np.array([0, 0, 0, 0, 1])
        }

        pot_odds = self.get_pot_odds(amount_to_play, pot)

        if amount_to_play == 0:
            return pot_odds_one_hot['free']
        elif pot_odds <= 0.25:
            return pot_odds_one_hot['cheap']
        elif pot_odds <= 0.5:
            return pot_odds_one_hot['moderate']
        elif pot_odds <= 0.75:
            return pot_odds_one_hot['expensive']
        else:
            return pot_odds_one_hot['very expensive']

    def get_bet_size(self, bet, initial_stack):
        """
        :param bet: Current sum of already invested chips in current round
        :param initial_stack: Number of chips handed to each player at the start of a round
        :return: Relative amount of chips already invested in the current round,
        compared to the maximal amount per round
        """
        return bet / initial_stack

    def get_bet_size_one_hot(self, bet, initial_stack):
        """
        :param bet: Current sum of already invested chips in current round
        :param initial_stack: Number of chips handed to each player at the start of a round
        :return: Relative amount of chips already invested in the current round,
        compared to the maximal amount per round. Represented by an one-hot encoding in ten percent steps
        """
        bet_size = self.get_bet_size(bet, initial_stack)
        return self._get_ten_percent_one_hot(bet_size)

    def get_aggressivity_one_hot(self, seats, own_uuid):
        """
        :param seats: All seats taking place in the poker game, not depending on their current status
        :param own_uuid: Uuid of the own player
        :return: One-hot encoded array of currently active opponents aggressivity in twenty percents steps.
        A players aggressivity is represented by the number of raises divided by the number of total actions
        """
        aggressivity_categories = {
            'very aggressive': np.array([1, 0, 0, 0, 0]),
            'aggressive': np.array([0, 1, 0, 0, 0]),
            'balanced': np.array([0, 0, 1, 0, 0]),
            'passive': np.array([0, 0, 0, 1, 0]),
            'very passive': np.array([0, 0, 0, 0, 1])
        }

        if not self.opponent_statistics_aggressivity and not self.opponent_statistics_tightness:
            self._create_opponent_statistics(seats, own_uuid)

        players = Players()
        active_players = players.get_active_players(seats, False, own_uuid)

        aggressivities = []
        for active_player in active_players:
            aggressivities.append(self.opponent_statistics_aggressivity[active_player]['aggressivity'])

        aggressivity_of_active_players = np.mean(aggressivities)
        if aggressivity_of_active_players > 0.8:
            return aggressivity_categories['very aggressive']
        elif aggressivity_of_active_players > 0.6:
            return aggressivity_categories['aggressive']
        elif aggressivity_of_active_players > 0.4:
            return aggressivity_categories['balanced']
        elif aggressivity_of_active_players > 0.2:
            return aggressivity_categories['passive']
        else:
            return aggressivity_categories['very passive']

    def get_tightness_one_hot(self, seats, own_uuid):
        """
        :param seats: All seats taking place in the poker game, not depending on their current status
        :param own_uuid: Uuid of the own player
        :return: One-hot encoded array of currently active opponents tightness in twenty percents steps.
        A players tightness is represented by the number of folds during the preflop street
        divided by the number of rounds that have been played already
        """
        tightness_categories = {
            'very tight': np.array([1, 0, 0, 0, 0]),
            'tight': np.array([0, 1, 0, 0, 0]),
            'balanced': np.array([0, 0, 1, 0, 0]),
            'loose': np.array([0, 0, 0, 1, 0]),
            'very loose': np.array([0, 0, 0, 0, 1])
        }

        tightness_of_active_players = self.get_tightness_of_active_players(seats, own_uuid)

        if tightness_of_active_players > 0.8:
            return tightness_categories['very tight']
        elif tightness_of_active_players > 0.6:
            return tightness_categories['tight']
        elif tightness_of_active_players > 0.4:
            return tightness_categories['balanced']
        elif tightness_of_active_players > 0.2:
            return tightness_categories['loose']
        else:
            return tightness_categories['very loose']

    def get_risk_factor(self, round_state, hole_card, own_uuid, debug_printouts):
        """
        :param round_state: Current round state
        :param hole_card: Hole cards of the own player
        :param own_uuid: Uuid of the own player
        :param debug_printouts: Parameter for debugging purpose only. Allows printing the calculated risk factor
        :return: Inverted hand rank, strengthened by late or weakened by early position
        """
        players = Players()
        number_of_active_players = len(players.get_active_players(round_state['seats'], False, own_uuid))
        hand_rank = self.get_hand_rank(hole_card, round_state, debug_printouts)
        [early_position, _, late_position] = self.get_position_one_hot(round_state, own_uuid)

        risk_factor = min(1, hand_rank * (len(round_state['seats']) / number_of_active_players))
        if early_position:
            risk_factor = risk_factor * 0.8
        elif late_position:
            risk_factor = risk_factor * 1.2

        risk_factor = min(1, risk_factor)
        risk_factor = 1 - risk_factor

        return risk_factor

    def get_risk_factor_one_hot(self, round_state, hole_card, own_uuid, debug_printouts):
        """
        :param round_state: Current round state
        :param hole_card: Hole cards of the own player
        :param own_uuid: Uuid of the own player
        :param debug_printouts: Parameter for debugging purpose only. Allows printing the calculated risk factor
        :return: Inverted hand rank, strengthened by late or weakened by early position.
        Represented by a one-hot encoding in twenty percent steps.
        """
        risk_categories = {
            'very low': [1, 0, 0, 0, 0],
            'low': [0, 1, 0, 0, 0],
            'medium': [0, 0, 1, 0, 0],
            'high': [0, 0, 0, 1, 0],
            'very high': [0, 0, 0, 0, 1]
        }

        risk_factor = self.get_risk_factor(round_state, hole_card, own_uuid, debug_printouts)

        if risk_factor > 0.8:
            return risk_categories['very low']
        elif risk_factor > 0.6:
            return risk_categories['low']
        elif risk_factor > 0.4:
            return risk_categories['medium']
        elif risk_factor > 0.2:
            return risk_categories['high']
        else:
            return risk_categories['very high']

    def get_player_statistics(self, seats, own_uuid):
        """
        :param seats: Participating players
        :param own_uuid: Own uuid, to exclude from dictionary
        :return: Returns a list, containing the following components for each opponent,
        starting with the player to our left, going clockwise:
        - is player active
        - aggressivity of player
        - tightness of player
        """
        if not self.opponent_statistics_aggressivity and not self.opponent_statistics_tightness:
            self._create_opponent_statistics(seats, own_uuid)

        statistics = []
        number_of_opponents = len(seats) - 1
        number_of_statistics_features = 3
        players = Players()
        index_of_own_player = players.get_index_of_player(seats, own_uuid, False)

        current_index = index_of_own_player
        while len(statistics) < number_of_opponents * number_of_statistics_features:
            current_index += 1

            if current_index > 3 * len(seats):
                str_index_of_own_player = index_of_own_player
                str_seats = seats
                raise IndexError('_get_player_statistics used to many iterations.\n'
                                 'index_own_player: ' + str(str_index_of_own_player) + '\n'
                                 'seats:' + str(str_seats))

            if seats[current_index % len(seats)]['uuid'] == own_uuid:
                continue

            statistics.append(0 if seats[current_index % len(seats)]['state'] == 'folded' else 1)
            try:
                statistics.append(self.opponent_statistics_aggressivity[seats[current_index % len(seats)]['uuid']]['aggressivity'])
                statistics.append(self.opponent_statistics_tightness[seats[current_index % len(seats)]['uuid']]['tightness'])
            except Exception as e:
                print(e)
                raise KeyError(
                    'KeyError at get_player_statistics:  ' +
                    'number_of_opponents: {}   '.format(number_of_opponents) +
                    'index_of_own_player: {}   '.format(index_of_own_player) +
                    'seats: {}   '.format(seats) +
                    'len(seats): {}   '.format(len(seats)) +
                    'statistics: {}   '.format(statistics) +
                    'self.opponent_statistics_tightness : {}   '.format(self.opponent_statistics_tightness) +
                    'self.opponent_statistics_aggressivity: {}   '.format(self.opponent_statistics_aggressivity) +
                    'What comes out with seats[current_index % len(seats): {}'.format(seats[current_index % len(seats)])
                )

        return statistics

    def get_tightness_of_active_players(self, seats, own_uuid):
        """
        :param seats: All seats taking place in the poker game, not depending on their current status
        :param own_uuid: Uuid of the own player
        :return: Mean tightness of currently active players.
        A players tightness is represented by the number of folds during the preflop street
        divided by the number of rounds that have been played already
        """
        if not self.opponent_statistics_aggressivity and not self.opponent_statistics_tightness:
            self._create_opponent_statistics(seats, own_uuid)

        players = Players()
        active_players = players.get_active_players(seats, False, own_uuid)

        tightness = []
        for active_player in active_players:
            tightness.append(self.opponent_statistics_tightness[active_player]['tightness'])

        return np.mean(tightness)

    def update_opponent_statistics(self, own_uuid: str, action: Dict[str, Union[str, int]], round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        No return value. The opponents statistics for aggressivity and tightness are being updated
        :param own_uuid: Uuid of the own player
        :param action: Last action, which was received in the receive_game_update_message method
        :param round_state: Current round state, containing street und seats
        """
        if action['player_uuid'] == own_uuid:
            return

        if len(self.opponent_statistics_aggressivity) == 0 and len(self.opponent_statistics_tightness) == 0:
            self._create_opponent_statistics(round_state['seats'], own_uuid)

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

    @staticmethod
    def _map_card_value_to_number(value):
        """
        :param value: Value of a card
        :return: Value of a card represented as number instead of string
        """
        value = value.upper()
        if value == 'T':
            return 10
        elif value == 'J':
            return 11
        elif value == 'Q':
            return 12
        elif value == 'K':
            return 13
        elif value == 'A':
            return 14
        else:
            return int(value)

    def _get_hole_card_rank(self, hole_card):
        """
        :param hole_card: hole card of own player
        :return: value between 0 and 1 depending on hole card strength. 1 represents the strongest hand, 0 the weakest
        """
        value_card_one = self._map_card_value_to_number(hole_card[0][0])
        value_card_two = self._map_card_value_to_number(hole_card[1][0])
        lower_value = str(min(value_card_one, value_card_two))
        higher_value = str(max(value_card_one, value_card_two))
        pair = value_card_one == value_card_two
        suited = hole_card[0][1] == hole_card[1][1]

        if pair:
            return 0.025 * self.hand_values_pairs[lower_value]
        elif suited:
            return 0.025 * self.hand_values_suited[lower_value][higher_value]
        else:
            return 0.025 * self.hand_values_offsuit[lower_value][higher_value]

    def _get_cards_rank(self, hole_card, round_state, debug_printouts):
        """
        :param hole_card: Hole cards of own player
        :param round_state: Current round state, containing community cards
        :param debug_printouts: Parameter for debugging purpose only. Allows printing the calculated hand rank
        :return: Float between 0 and 1, representing the current five card rank among all possible poker hands.
        0 represents the weakest five card combination, 1 the strongest (Royal Flush)
        """
        evaluator = Evaluator()

        board = []
        hand = []
        if len(round_state['community_card']) >= len(board):
            for card in round_state['community_card']:
                board.append(Card.new(card))

        for card in hole_card:
            hand.append(Card.new(card))

        score = evaluator.evaluate(board, hand)

        if debug_printouts:
            Card.print_pretty_cards(board + hand)
            print(Card.print_pretty_cards(board + hand))

        return 1 - evaluator.get_five_card_rank_percentage(score)

    def _create_opponent_statistics(self, seats, own_uuid) -> None:
        """
        Creates two empty dictionaries to store information about opponent behaviour
        One dictionary stores information about the opponents aggressivty, the other about tightness
        :param seats: Participating players
        :param own_uuid: Own uuid, to exclude from dictionary
        """
        for seat in seats:
            if seat['uuid'] == own_uuid:
                continue

            self.opponent_statistics_aggressivity[seat['uuid']] = {
                'number_of_raises': 0,
                'number_of_actions': 0,
                'aggressivity': 0
            }

            self.opponent_statistics_tightness[seat['uuid']] = {
                'hands_folded': 0,
                'tightness': 0
            }

    def _get_ten_percent_one_hot(self, percentage):
        """
        :param percentage: Percentage value, which needs to be one-hot encoded
        :return: One-hot encoding of the given percentage, represented by ten percent steps in a binary array
        """
        if percentage >= 0.9:
            return self.ten_percent_one_hot['90%-100%']
        elif percentage >= 0.8:
            return self.ten_percent_one_hot['80%-90%']
        elif percentage >= 0.7:
            return self.ten_percent_one_hot['70%-80%']
        elif percentage >= 0.6:
            return self.ten_percent_one_hot['60%-70%']
        elif percentage >= 0.5:
            return self.ten_percent_one_hot['50%-60%']
        elif percentage >= 0.4:
            return self.ten_percent_one_hot['40%-50%']
        elif percentage >= 0.3:
            return self.ten_percent_one_hot['30%-40%']
        elif percentage >= 0.2:
            return self.ten_percent_one_hot['20%-30%']
        elif percentage >= 0.1:
            return self.ten_percent_one_hot['10%-20%']
        else:
            return self.ten_percent_one_hot['0%-10%']

    def _get_five_percent_one_hot(self, percentage):
        """
        :param percentage: Percentage value, which needs to be one-hot encoded
        :return: One-hot encoding of the given percentage, represented by five percent steps in a binary array
        """
        five_percent_one_hot = {
            '0%-5%': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '5%-10%': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '10%-15%': np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '15%-20%': np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '20%-25%': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '25%-30%': np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '30%-35%': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '35%-40%': np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '40%-45%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '45%-50%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '50%-55%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            '55%-60%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            '60%-65%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            '65%-70%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            '70%-75%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            '75%-80%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            '80%-85%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            '85%-90%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            '90%-95%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            '95%-100%': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        }

        if percentage >= 0.95:
            return five_percent_one_hot['95%-100%']
        elif percentage >= 0.9:
            return five_percent_one_hot['90%-95%']
        elif percentage >= 0.85:
            return five_percent_one_hot['85%-90%']
        elif percentage >= 0.8:
            return five_percent_one_hot['80%-85%']
        elif percentage >= 0.75:
            return five_percent_one_hot['75%-80%']
        elif percentage >= 0.7:
            return five_percent_one_hot['70%-75%']
        elif percentage >= 0.65:
            return five_percent_one_hot['65%-70%']
        elif percentage >= 0.6:
            return five_percent_one_hot['60%-65%']
        elif percentage >= 0.55:
            return five_percent_one_hot['55%-60%']
        elif percentage >= 0.5:
            return five_percent_one_hot['50%-55%']
        elif percentage >= 0.45:
            return five_percent_one_hot['45%-50%']
        elif percentage >= 0.4:
            return five_percent_one_hot['40%-45%']
        elif percentage >= 0.35:
            return five_percent_one_hot['35%-40%']
        elif percentage >= 0.3:
            return five_percent_one_hot['30%-35%']
        elif percentage >= 0.25:
            return five_percent_one_hot['25%-30%']
        elif percentage >= 0.2:
            return five_percent_one_hot['20%-25%']
        elif percentage >= 0.15:
            return five_percent_one_hot['15%-20%']
        elif percentage >= 0.1:
            return five_percent_one_hot['10%-15%']
        elif percentage >= 0.05:
            return five_percent_one_hot['5%-10%']
        else:
            return five_percent_one_hot['0%-5%']
