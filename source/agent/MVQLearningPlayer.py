import random
from typing import Dict, List, Union, Tuple

from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

"""
Bot RE1 Qlearning - deployed to AICrowd 31.03.2020
Bot contains lot of comments. Shows, how the team struggled with python as well as the concept of the model at the 
beginning. 
"""


class MVQLearningPlayer(BasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    __number_of_simulations = 100

    __next_street = {
        'preflop': 'flop',
        'flop': 'turn',
        'turn': 'river',
        'river': 'showdown',
        'showdown': 'preflop'
    }

    __previous_street = {
        'flop': 'preflop',
        'turn': 'flop',
        'river': 'turn',
        'showdown': 'river'
    }

    def __str__(self):
        return "MV Q-Learning"

    def __init__(self):
        self.__qtable = {
            'preflop': {
                'call': 0,
                'raise': 0,
                'fold': 0
            },
            'flop': {
                'call': 0,
                'raise': 0,
                'fold': 0
            },
            'turn': {
                'call': 0,
                'raise': 0,
                'fold': 0
            },
            'river': {
                'call': 0,
                'raise': 0,
                'fold': 0
            },
            'showdown': {
                'call': 0,
                'raise': 0,
                'fold': 0
            }
        }

        self.__reward_table = {}
        self.__hole_cards = []
        self.__last_action = ''
        self.__mybet = 0
        self.__stack_size = 0
        # self.__reward_points = 0
        self.__epsilon_explore = 0.15
        self.__is_first_action_of_street = True
        self.__first_action_of_street = ''
        self.__estimated_reward_from_first_action_of_street = 0
        self.__previous_winrate = 0

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:

        # t1 = int(round(time.time() * 1000))
        # Entry for QTable

        street_state = round_state['street']
        win_rate = self._get_win_rate(round_state)
        pot = round_state['pot']['main']['amount']
        my_stack = self.__stack_size - self.__mybet

        # Update Reward Table basierend auf neue win_rate, pot und my_stack Werte
        self.__update_reward(win_rate, pot, my_stack)

        possible_moves = []
        for dict in valid_actions:
            possible_moves.append(dict["action"])

        valid_q_actions = {k: v for k, v in self.__qtable[street_state].items() if k in possible_moves}

        if random.uniform(0, 1) < self.__epsilon_explore:
            action = random.choice(list(valid_q_actions.keys()))
        else:
            action = max(valid_q_actions, key=valid_q_actions.get)

        amount = 0

        #   print('Our action from qtabel is: ' + str(action))
        if action == 'raise':
            # Zufälligen Betrag im erlaubten Bereich zum raisen wählen
            action = valid_actions[2]['action']  # fetch RAISE action info
            amount_limits = valid_actions[2]['amount']
            # amount kann -1 sein wenn nicht mehr geraised werden kann, dann muss gecallt werden
            if amount == -1:
                amount = valid_actions[1]['amount']
            else:
                # Berechne raise basierend auf Gewinnwahrscheinlichkeit
                range = max(amount_limits["min"], amount_limits["max"]) - amount_limits["min"]
                amount = amount_limits["min"] + 2 * win_rate * range
                # amount = random.randint(amount_limits["min"], max(amount_limits["min"], amount_limits["max"]))

        if action == 'call' or amount > valid_actions[1]['amount']:
            amount = valid_actions[1]['amount']

        # Vorübergehend: Fix für random flops bei guter Win-Rate
        if action == 'fold':
            if win_rate > 2 / len(round_state['seats']):
                action = 'call'

        self.__mybet += amount
        self.__last_action = action
        # print('Our action is: ' + str(action))
        # print('Amount set is: '+ str(amount))

        # Erste Aktion und erwarteten Gewinn pro street merken, um später Q-Table nachzuführen
        # if self.__is_first_action_of_street:
        #     self.__first_action_of_street = action
        #     self.__estimated_reward_from_first_action_of_street = win_rate * pot
        #     self.__is_first_action_of_street = False

        # t2 = int(round(time.time() * 1000))
        # print(str(t2-t1))

        self.__update_qtable(round_state, False)
        self.__previous_winrate = win_rate
        return action, amount

    def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
        """
        Called once the game started.
        :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
        """
        self.__stack_size = game_info['rule']['initial_stack']
        self.__game_finished = False

    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict[str, Union[str, int]]]) -> None:
        """
        Called once a round starts.
        :param round_count: Round number, in Cash Game always 1.
        :param hole_card: Cards in possession of the player.
        :param seats: Players at the table.
        """
        self.__hole_cards = hole_card

    def receive_street_start_message(self, street: str, round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called at every stage (preflop, flop, turn, river, showdown).
        :param street: Game stage
        :param round_state: Dictionary containing the round state
        """

        # Idee: Measure reward anhand von gewählter Action
        #   if street == 'preflop': return

        # pot = round_state['pot']['main']['amount']
        # winrate = self._get_win_rate(round_state)

        # self.__update_qtable(round_state)

        #  self.__is_first_action_of_street = True

        pass

    def receive_game_update_message(self, action: Dict[str, Union[str, int]],
                                    round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called after every action made by any of the players.
        :param action: Dict containing the player uuid and the executed action
        :param round_state: Dictionary containing the round state
        """

        # current_street = [round_state['street']][0]
        # if round_state['action_histories'][current_street][-1]['uuid'] != self.uuid:
        #     self._update_reward(action, round_state)
        #     return
        #
        # if not self.__last_action:
        #     return
        #
        # self._update_qtable(round_state)

    def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                                     hand_info: [List[Dict[str, Union[str, Dict]]]],
                                     round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        won = False
        self.__game_finished = True

        for winner in winners:
            if winner['uuid'] == self.uuid:
                won = True
                self.__update_qtable(round_state, won)

        print(won)
        print(self.__qtable)
        print(self.__reward_table)

    def __update_qtable(self, round_state, won) -> None:

        current_street = [round_state['street']][0]
        next_street = self.__next_street[current_street]
        #    pot = round_state['pot']['main']['amount']

        alpha = 0.01
        y = 0.9

        #  print(current_street)
        max_q = max(self.__qtable[next_street].values())

        if won:
            self.__qtable[current_street][self.__last_action] =  self.__qtable[current_street][
                                                                    self.__last_action] + alpha * (
                                                                        self.__reward_table['won'] + y
                                                                        * max_q - self.__qtable[current_street][
                                                                            self.__last_action])
        elif self.__game_finished:
            self.__qtable[current_street][self.__last_action] =   self.__qtable[current_street][
                                                                    self.__last_action] + alpha * (
                                                                        self.__reward_table['lost'] + y
                                                                        * max_q - self.__qtable[current_street][
                                                                            self.__last_action])
        else:
            self.__qtable[current_street][self.__last_action] = self.__qtable[current_street][
                                                                    self.__last_action] + alpha * (
                                                                        self.__reward_table[self.__last_action] + y
                                                                        * max_q - self.__qtable[current_street][
                                                                            self.__last_action])
        # print(self.__qtable)

    def __update_reward(self, win_rate, pot, stack):
        self.__reward_table = {
            'call': win_rate * pot,
            'raise': (win_rate - self.__previous_winrate) * pot,
            'fold': (1 - win_rate) * stack,
            'won': pot + stack,
            'lost': 0
        }

    def _get_win_rate(self, round_state):
        # Gewinn Wahrscheinlichkeit anhand von Monte Carlo Simulation berechnen
        nb_player = len(round_state['seats'])
        community_card = round_state['community_card']

        win_rate = estimate_hole_card_win_rate(
            nb_simulation=self.__number_of_simulations,
            nb_player=nb_player,
            hole_card=gen_cards(self.__hole_cards),
            community_card=gen_cards(community_card)
        )
        return win_rate
