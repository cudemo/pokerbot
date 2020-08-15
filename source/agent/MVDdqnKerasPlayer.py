import configparser
import os
from typing import Dict, List, Union, Tuple

from agent.MVBasePokerPlayer import MVBasePokerPlayer
from components.DdqnKerasModel import DdqnKerasModel
from components.Rewards import Rewards

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MVDdqnKerasPlayer(MVBasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    def __init__(self, player_name="DdqnKerasPlayer",reward_system=1, *args, **kwargs):
        self.player_name = player_name
        super(MVDdqnKerasPlayer, self).__init__(*args, **kwargs)

        # Network Parameters which will be used
        self.out_layer_size = 4
        self.agent = DdqnKerasModel(name=player_name, alpha=0.01, gamma=1, input_layer_size=self.input_layer_size,
                                    number_of_parameters=self.number_of_parameters, out_layer_size=self.out_layer_size,
                                    memory_size=10000, batch_size=32)

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
        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        self.episode_printout = config['log_settings'].getboolean('episode_printout_ddqn_keras')

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
