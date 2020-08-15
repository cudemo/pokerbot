from typing import Dict, List, Union, Tuple

from pypokerengine.players import BasePokerPlayer

"""
Default MyBotPlayer provided by PyPokerEngine
"""
class MyBotPlayer(BasePokerPlayer):
    """
    Documentation for callback arguments given here:
    https://github.com/ishikota/PyPokerEngine/blob/master/AI_CALLBACK_FORMAT.md
    """

    def __str__(self):
        return "MyBot"

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
        """
        Define what action the player should execute.
        :param valid_actions: List of dictionary containing valid actions the player can execute.
        :param hole_card: Cards in possession of the player encoded as a list of strings.
        :param round_state: Dictionary containing relevant information and history of the game.
        :return: action: str specifying action type. amount: int action argument.
        """
        call_action_info = valid_actions[1]
        action, amount = call_action_info["action"], call_action_info["amount"]
        return action, amount

    def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
        """
        Called once the game started.
        :param game_info: Dictionary containing game rules, # of rounds, initial stack, small blind and players at the table.
        """
        pass

    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict[str, Union[str, int]]]) -> None:
        """
        Called once a round starts.
        :param round_count: Round number, in Cash Game always 1.
        :param hole_card: Cards in possession of the player.
        :param seats: Players at the table.
        """
        pass

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
        pass

    def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                                     hand_info: [List[Dict[str, Union[str, Dict]]]],
                                     round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Called at the end of the round.
        :param winners: List of the round winners containing the stack and player information.
        :param hand_info: List containing a Dict for every player at the table describing the players hand this round.
        :param round_state: Dictionary containing the round state
        """
        pass
