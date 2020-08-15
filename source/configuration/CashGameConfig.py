import os
from datetime import datetime
from typing import Dict
import locale

from baseline import *
from agent import MVDqnPlayer, MVDdqnKerasPlayer
from agent import MVDdqnPlayer
from agent import MVDdqnPerPlayer

from pypokerengine.api.game import setup_config, start_poker

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


class CashGameConfig():

    def __init__(self, small_blind_amount: int = 1, evaluations: int = 1000000, log_file_location: str = './logs'):
        """
        Setup a new CashGame
        :param small_blind_amount: int. Specify small blind
        :param evaluations: int. Number of poker hands to evaluate
        :param log_file_location: str. location for a detailed log file.
        """
        now = datetime.strftime(datetime.now(), "%d.%m.%Y-%H:%M:%S")
        self.log_file_location = os.path.join(log_file_location, f"evaluation_{now}.json")
        self.initial_stack = 100 * small_blind_amount * 2
        self.config = setup_config(max_round=evaluations, initial_stack=self.initial_stack,
                                   small_blind_amount=small_blind_amount)
        self.evaluations = evaluations
        self.player_final_stack = {}
        self.baselines = [BaselinePokerPlayer.BaselinePlayer, CallBaselinePokerPlayer.CallBaselinePlayer,
                          RandomPokerPlayer.RandomPlayer]

    def run_evaluation(self, verbose: int = 0) -> Dict[str, int]:
        if len(self.config.players_info) <= 1:
            raise AssertionError("At least two players have to be seated.")

        game_result = start_poker(self.config, verbose=verbose, cashgame=True, log_file_location=self.log_file_location)

        # Prettify game result:
        for player in game_result['players']:
            self.player_final_stack[player['name']] = player['cashgame_stack']
        self.player_final_stack = {k: v for k, v in
                                   sorted(self.player_final_stack.items(), key=lambda item: item[1], reverse=True)}

        # Print out game result
        for rank, (name, stack) in enumerate(self.player_final_stack.items()):
            print(f"{rank + 1:2}. Player: {name:>25}, Stack: {stack:n}")

        return self.player_final_stack

    def register_player(self, name: str, algorithm):
        self.config.register_player(name=name, algorithm=algorithm)

    def _get_config(self):
        return self.config

    def add_all_available_baselines(self):
        for baseline in self.baselines:
            self.config.register_player(name=str(baseline), algorithm=baseline())

    def add_dqn_bots(self):
        """
        Register DQN bots to be used for evaluation
        reward_system 1: Simple
        reward_system 2: Complex
        """

        player_name = 'DdqnPerPlayer_1'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=True, with_per=True,
                                                     reward_system=1, play_blinds=True)
        self.config.register_player(name=player_name, algorithm=dqn_player)

        player_name = 'DdqnPerPlayer_2'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=True, with_per=True,
                                                     reward_system=1, play_blinds=True)
        self.config.register_player(name=player_name, algorithm=dqn_player)

        player_name = 'DdqnPerPlayer_3'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=True, with_per=True,
                                                     reward_system=1, play_blinds=False)
        self.config.register_player(name=player_name, algorithm=dqn_player)

        player_name = 'DdqnPlayer_4'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=False, with_per=False,
                                                     reward_system=2, play_blinds=False)
        self.config.register_player(name=player_name, algorithm=dqn_player)

        player_name = 'DdqnPerPlayer_5'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=False, with_per=True,
                                                     reward_system=1, play_blinds=False)
        self.config.register_player(name=player_name, algorithm=dqn_player)

        player_name = 'DdqnPerPlayer_6'
        dqn_player = MVDdqnPerPlayer.MVDdqnPerPlayer(player_name=player_name, split_network=False, with_per=True,
                                                     reward_system=1, play_blinds=False)
        self.config.register_player(name=player_name, algorithm=dqn_player)
