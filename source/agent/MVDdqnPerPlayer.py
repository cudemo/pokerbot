import configparser
import os
from typing import Dict, List, Union, Tuple

import tensorflow as tf

from agent.MVBasePokerPlayer import MVBasePokerPlayer
from components.DdqnPerSplitModel import DdqnPerSplitModel
from components.Rewards import Rewards

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class MVDdqnPerPlayer(MVBasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    def __init__(self, player_name="DdqnPerPlayer", split_network=True, with_per=True, reward_system=1,
                 play_blinds=False, *args, **kwargs):

        self.player_name = player_name
        super(MVDdqnPerPlayer, self).__init__(*args, **kwargs)

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')

        # Network Parameters which will be used
        self.out_layer_size = 4
        self.split_network = split_network
        self.first_action_after_preflop = False
        self.play_blinds = play_blinds

        if split_network:
            self.input_layer_size = 156
            self.number_of_parameters = 156
            self.input_layer_preflop_size = 52
        else:
            self.input_layer_preflop_size = 0

        self.agent = DdqnPerSplitModel(name=player_name, alpha=0.05, gamma=1, input_layer_size=self.input_layer_size,
                                       input_layer_preflop_size=self.input_layer_preflop_size,
                                       out_layer_size=self.out_layer_size, split_network=split_network,
                                       with_per=with_per, memory_size=10000, batch_size=128)

        # Reward Systems
        self.rewards = Rewards()
        self.reward_system = reward_system
        if reward_system == 1:
            self.reward_model_used = 'Simple'
        elif reward_system == 2:
            self.reward_model_used = 'Complex'
        else:
            self.reward_model_used = 'Default Simple'

        self.round_actions = []

        # Printouts - set all to False before uploading
        self.episode_printout = config['log_settings'].getboolean('episode_printout_ddqnper')

    def __str__(self):
        return self.player_name

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
        """
        Define what action the player should execute.
        :param valid_actions: List of dictionary containing valid actions the player can execute.
        :param hole_card: Cards in possession of the player encoded as a list of strings.
        :param round_state: Dictionary containing relevant information and history of the game.
        :return: action: str specifying action type. amount: int action argument.
        """
        return super().declare_action(valid_actions, hole_card, round_state)

    def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
        """
        Called once the game started.
        :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
        """
        super().receive_game_start_message(game_info)

    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict[str, Union[str, int]]]) -> None:
        """
        Called once a round starts.
        :param round_count: Round number, in Cash Game always 1.
        :param hole_card: Cards in possession of the player.
        :param seats: Players at the table.
        """
        super().receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street: str, round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called at every stage (preflop, flop, turn, river, showdown).
        :param street: Game stage
        :param round_state: Dictionary containing the round state
        """
        super().receive_street_start_message(street, round_state)

    def receive_game_update_message(self, action: Dict[str, Union[str, int]],
                                    round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Gets called after every action made by any of the players.
        :param action: Dict containing the player uuid and the executed action
        :param round_state: Dictionary containing the round state
        """
        super().receive_game_update_message(action, round_state)

    def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                                     hand_info: [List[Dict[str, Union[str, Dict]]]],
                                     round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Called at the end of the round.
        :param winners: List of the round winners containing the stack and player information.
        :param hand_info: List containing a Dict for every player at the table describing the players hand this round.
        :param round_state: Dictionary containing the round state
        """
        super().receive_round_result_message(winners, hand_info, round_state)
