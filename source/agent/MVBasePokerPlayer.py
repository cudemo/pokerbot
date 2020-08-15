import configparser
import smtplib
import time
from email.message import EmailMessage
from typing import Dict, Union, List, Tuple

import numpy as np
from pypokerengine.players import BasePokerPlayer

from components.Features import Features
from components.Players import Players
from components.Plots import Plots, ActionData, WonAmountData, CashgameStackData, LostAmountData


class MVBasePokerPlayer(BasePokerPlayer):
    def __init__(self, *args, **kwargs):
        super(MVBasePokerPlayer, self).__init__(*args, **kwargs)

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')

        # Rules
        self.initial_stack = 0
        self.rounds_to_play = 0
        self.small_blind_amount = 0

        # Round state
        self.last_action_index = False
        self.last_state = []
        self.preflop_allin = False
        self.folded = False
        self.round_count = 0
        self.total_bet = 0
        self.amount_to_play = 0
        self.risk_factor = 0
        self.hand_rank = 0
        self.agent = 0
        self.split_network = False  # Used for preflop network split in MVDdqnPer
        self.play_blinds = False

        # Statistics
        self.cashgame_stack = 0
        self.times_won = 0
        self.times_folded = 0
        self.times_small_called = 0
        self.times_hr_raised = 0
        self.times_big_called = 0
        self.times_all_in = 0
        self.preflop_allin_counter = 0
        self.high_bet_fold_counter = 0
        self.folded_in_row_counter = 0
        self.low_hand_rank_fold_counter = 0

        # Features
        self.features = Features()
        self.won_amount_history = []
        self.lost_amount_history = []
        self.round_actions_length = []

        # Neural Network
        self.input_layer_size = 66
        self.number_of_parameters = 66

        # Printouts - set all to False before uploading
        self.debug_printouts = config['log_settings'].getboolean('debug_printouts')
        self.measure_declare_action_time = config['log_settings'].getboolean('measure_declare_action_time')
        self.send_statistics_mail = config['log_settings'].getboolean('send_statistics_mail')
        self.aicrowd_evaluation = config['log_settings'].getboolean('aicrowd_evaluation')

        # Plots
        plots_enabled = config['plot_settings'].getboolean('plots_enabled')
        self.plot_cashgame_stack = plots_enabled and config['plot_settings'].getboolean('plot_cashgame_stack')
        if self.plot_cashgame_stack:
            self.cashgame_stack_data = CashgameStackData()
            Plots.get_instance().register_agent_cashgame_stack(str(self))

        self.plot_won_amount = plots_enabled and config['plot_settings'].getboolean('plot_won_amount')
        if self.plot_won_amount:
            self.won_amount_data = WonAmountData()
            Plots.get_instance().register_agent_won_amount(str(self))

        self.plot_lost_amount = plots_enabled and config['plot_settings'].getboolean('plot_lost_amount')
        if self.plot_lost_amount:
            self.lost_amount_data = LostAmountData()
            Plots.get_instance().register_agent_lost_amount(str(self))

        self.plot_actions = plots_enabled and config['plot_settings'].getboolean('plot_actions')
        if self.plot_actions:
            self.action_data = ActionData()
            Plots.get_instance().register_agent_actions(str(self))

    def declare_action(self, valid_actions: List[Dict[str, Union[int, str]]], hole_card: List[str],
                       round_state: Dict[str, Union[int, str, List, Dict]]) -> Tuple[Union[int, str], Union[int, str]]:
        if self.measure_declare_action_time:
            t1 = int(round(time.time() * 1000))

        self.hand_rank = self.features.get_hand_rank(hole_card, round_state, self.debug_printouts)
        self.risk_factor = self.features.get_risk_factor(round_state, hole_card, self.uuid, self.debug_printouts)
        self.amount_to_play = valid_actions[1]['amount']
        tightness = self.features.get_tightness_of_active_players(round_state['seats'], self.uuid)
        use_network = self._use_preflop_statistic_network_decision(round_state, self.amount_to_play, self.hand_rank,
                                                                   self.play_blinds)
        state = self.get_state(round_state, hole_card, round_state['street'])

        if use_network:
            predicted_action = self.agent.choose_action(state, round_state['street'])
        else:
            predicted_action = 0
            self.low_hand_rank_fold_counter += 1

        total_bet_old = self.total_bet
        action, total_bet_new, action_index = self._validate_action(predicted_action, valid_actions)
        amount_added = total_bet_new - total_bet_old

        if amount_added < 0:  # if we fold, amount_added could be negative
            amount_added = 0

        self.total_bet += amount_added

        # Remember state transition variables for round_result to fill MemoryBuffer
        if self.split_network and self.first_action_after_preflop:
            self.round_actions.append(
                [[], self.last_action_index, state, action_index, amount_added, self.blind_paid,
                 self.amount_to_play, round_state['street'], self.risk_factor, self.hand_rank, tightness, 0])
            self.first_action_after_preflop = False
        else:
            self.round_actions.append(
                [self.last_state, self.last_action_index, state, action_index, amount_added,
                 self.blind_paid, self.amount_to_play, round_state['street'], self.risk_factor, self.hand_rank,
                 tightness, 0])
            self.blind_paid = 0

        # Remember decision and corresponding state for next round remember function
        self.last_action_index = action_index
        self.last_state = state

        self.preflop_allin = True if round_state['street'] == 'preflop' \
                                     and total_bet_new == self.initial_stack else False

        if self.debug_printouts:
            print('Feeding state: {} into model for prediction'.format(state))
            print('Action: {}, Amount: {}'.format(action, total_bet_new))

        if self.measure_declare_action_time:
            t2 = int(round(time.time() * 1000))
            print('Decision taken in: {} seconds'.format((t2 - t1) / 1000))

        if self.plot_actions:
            self.action_data.add_action(action)

        return action, total_bet_new

    def receive_game_start_message(self, game_info: Dict[str, Union[int, Dict, List]]) -> None:
        self.initial_stack = game_info['rule']['initial_stack']
        self.small_blind_amount = game_info['rule']['small_blind_amount']
        self.rounds_to_play = game_info['rule']['max_round']
        self._send_start_mail(game_info['rule']['max_round'])

        if self.plot_won_amount:
            self.won_amount_data.set_total_rounds(self.rounds_to_play)
        if self.plot_actions:
            self.action_data.set_total_rounds(self.rounds_to_play)
        if self.plot_cashgame_stack:
            self.cashgame_stack_data.set_total_rounds(self.rounds_to_play)

    def receive_game_update_message(self, action: Dict[str, Union[str, int]],
                                    round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        self.features.update_opponent_statistics(self.uuid, action, round_state)

    def receive_round_start_message(self, round_count: int, hole_card: List[str],
                                    seats: List[Dict[str, Union[str, int]]]) -> None:
        self._reset_state(round_count)
        self.hole_cards = hole_card
        self.first_action_after_preflop = False

    def receive_street_start_message(self, street: str, round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        if street == 'flop' and not self.first_action_after_preflop:
            self.first_action_after_preflop = True  # split network variable needed

        if round_state['street'] == 'preflop':
            players = Players()
            if players.get_index_of_player(round_state['seats'], self.uuid, False) == round_state['small_blind_pos']:
                self.blind_paid = self.small_blind_amount
                self.total_bet += self.blind_paid
            elif players.get_index_of_player(round_state['seats'], self.uuid, False) == round_state['big_blind_pos']:
                self.blind_paid = 2 * self.small_blind_amount
                self.total_bet += self.blind_paid

    def receive_round_result_message(self, winners: List[Dict[str, Union[int, str]]],
                                     hand_info: [List[Dict[str, Union[str, Dict]]]],
                                     round_state: Dict[str, Union[int, str, List, Dict]]) -> None:
        """
        Called at the end of the round.
        :param winners: List of the round winners containing the stack and player information.
        :param hand_info: List containing a Dict for every player at the table describing the players hand this round.
        :param round_state: Dictionary containing the round state
        """
        self.cashgame_stack -= self.total_bet
        won, won_amount = self._get_won_and_amount(winners)

        reward = 0
        accumulated_reward = 0
        round_actions_length = len(self.round_actions)

        if round_actions_length > 0:
            self.round_actions_length.append(round_actions_length)
            try:
                # set done flag of last action to 1
                self.round_actions[-1][-1] = 1
            except Exception as e:
                print(e)

            for [last_state, last_action_index, state, action_index, amount_added, blind_paid,
                 minimum_call_amount, street, risk_factor, hand_rank, tightness, done] in self.round_actions:

                if self.reward_system == 2:
                    reward = self.rewards.get_reward_complex(done, won, won_amount, self.total_bet, amount_added,
                                                             action_index, minimum_call_amount, self.initial_stack,
                                                             self.folded, street, risk_factor, hand_rank, tightness,
                                                             blind_paid)
                else:
                    reward = self.rewards.get_reward_simple(done, won, won_amount, amount_added, self.folded,
                                                            blind_paid)

                # caused by transition, blind_paid would be subtracted two times if we lost the game. this is due
                # to the fact that we want to count blinds in memory buffer for both cases: only one round action
                # (done Flag = 1) and more than ond round action, therefore last_action = [] transition would be ignored
                if last_state != [] and blind_paid > 0 and not won:
                    accumulated_reward += reward + blind_paid
                else:
                    accumulated_reward += reward

                if done:
                    # State transition for terminal state irrelevant for Q-learning, as only reward will be used
                    self.agent.remember(state, action_index, reward, state, done, street)
                elif last_state == []:
                    # nothing to remember for first action, as state transition not happened
                    # blind will be remembered in next transition, as we have more than two round_actions, we have a
                    # full transition to observe
                    continue
                else:
                    self.agent.remember(last_state, last_action_index, reward, state, done, street)

        self.agent.replay()

        # Update counters
        self._update_statistic_counters(won)

        self._print_episode(accumulated_reward, hand_info, round_state, won)

        if self.plot_cashgame_stack:
            self.cashgame_stack_data.add_cashgame_stack(self.cashgame_stack, self.round_count)
        self._update_agent_statistics(won, won_amount, self.total_bet)

        if self.rounds_to_play == self.round_count:
            self.agent.save_model()
            average_lost_amount, average_won_amount, total_lost_amount, total_won_amount, average_round_actions_amount = self._statistics_printouts()
            if self.send_statistics_mail:
                self._send_statistics_mail(len(round_state['seats']), round_state, average_lost_amount,
                                           average_won_amount, total_lost_amount, total_won_amount,
                                           average_round_actions_amount)
            self._plot_agent_statistics()

    def get_state(self, round_state, hole_card, street):
        pot_size_one_hot = self.features.get_pot_size_one_hot(round_state['pot']['main']['amount'],
                                                              len(round_state['seats']), self.initial_stack)
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(self.amount_to_play, round_state['pot']['main']['amount'])
        bet_size_one_hot = self.features.get_bet_size_one_hot(self.total_bet, self.initial_stack)
        number_of_opponents_one_hot = self.features.get_number_of_opponents_one_hot(round_state['seats'], self.uuid)
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)
        hand_rank_one_hot = self.features.get_hand_rank_one_hot(hole_card, round_state, self.debug_printouts)

        if self.split_network and street == 'preflop':
            state = position_one_hot, number_of_opponents_one_hot, pot_odds_one_hot, \
                    pot_size_one_hot, bet_size_one_hot, hand_rank_one_hot

        elif not self.split_network:
            street_one_hot = self.features.get_street_one_hot(round_state['street'])
            aggressivity_one_hot = self.features.get_aggressivity_one_hot(round_state['seats'], self.uuid)
            tightness_one_hot = self.features.get_tightness_one_hot(round_state['seats'], self.uuid)

            state = position_one_hot, street_one_hot, number_of_opponents_one_hot, pot_odds_one_hot, \
                    aggressivity_one_hot, tightness_one_hot, pot_size_one_hot, bet_size_one_hot, hand_rank_one_hot

        else:
            street_one_hot = self.features.get_street_one_hot(round_state['street'])
            hand_probabilities_histogram_one_hot = \
                self.features.get_hand_probabilities_histogram_one_hot(hole_card,round_state['community_card'])

            state = position_one_hot, street_one_hot, number_of_opponents_one_hot, pot_odds_one_hot, \
                    hand_probabilities_histogram_one_hot, pot_size_one_hot, bet_size_one_hot, hand_rank_one_hot

        state = np.concatenate(state)
        return state

    def _reset_state(self, round_count):
        """
        Reset the round state at the beginning of very round.
        :param round_count: Update round information
        :return: nothing
        """
        self.round_actions = []
        self.last_action_index = False
        self.last_state = []
        self.preflop_allin = False
        self.folded = False
        self.round_count = round_count
        self.total_bet = 0
        self.amount_to_play = 0
        self.risk_factor = 0
        self.hand_rank = 0
        self.blind_paid = 0

    def _update_statistic_counters(self, won):
        # If put > 50 and folded afterwards, count as high bet fold
        high_bet_fold_threshold = self.initial_stack * 0.25

        if self.total_bet == self.initial_stack:
            self.times_all_in += 1
        if won:
            self.times_won += 1
            if self.episode_printout:
                print("WON")
            self.folded_in_row_counter = 0
        elif not self.folded:
            self.folded_in_row_counter = 0
        else:
            self.folded_in_row_counter += 1
        if self.preflop_allin:
            self.preflop_allin_counter += 1
        if self.folded and self.total_bet >= high_bet_fold_threshold:
            self.high_bet_fold_counter += 1

    def _get_won_and_amount(self, winners):
        won = 0
        won_amount = 0
        if not self.folded:
            for winner in winners:
                if winner['uuid'] == self.uuid:
                    won = 1
                    won_amount = winner['stack'] - self.initial_stack
                    self.cashgame_stack = winner['cashgame_stack'] + won_amount
                    self.won_amount_history.append(won_amount)
                    break
        if not won:
            self.lost_amount_history.append(self.total_bet)

        return won, won_amount

    def _validate_action(self, predicted_action, valid_actions):
        """
        Validate if chosen action is valid action.
        Set amount based on chosen action
        :param predicted_action: action_index to take
        :param valid_actions: valid actions given by PyPokerEngine
        :return:
        """
        action = ''
        amount = 0

        action_types = ['fold', 'small_call', 'big_call', 'hr_raise']
        action_type = action_types[predicted_action]
        action_index = 0

        if action_type == 'hr_raise':
            action = valid_actions[2]['action']  # fetch RAISE action info
            action_index = 2
            amount_limits = valid_actions[2]['amount']

            amount = amount_limits["min"]

            # amount can be -1 if it is no longer possible to raise, must call instead
            if amount == -1:
                amount = valid_actions[1]['amount']
            elif amount < self.initial_stack * self.hand_rank:
                amount = np.floor(self.initial_stack * self.hand_rank)

            if amount > amount_limits["max"]:
                amount = amount_limits["max"]

            if self.debug_printouts:
                print('Amount set to: ', amount)

            self.times_hr_raised += 1

        elif action_type == 'small_call':
            amount = valid_actions[1]['amount']

            # We want to call half of our remaining stack and declare this a "small call"
            stack = self.initial_stack - self.total_bet
            small_call_threshold = self.total_bet + (0.5 * stack)

            if amount > small_call_threshold:
                action_type = 'fold'
            else:
                action = valid_actions[1]['action']
                action_index = 1
                self.times_small_called += 1

        elif action_type == 'big_call':
            action = valid_actions[1]['action']
            action_index = 1
            amount = valid_actions[1]['amount']
            self.times_big_called += 1

        if action_type == 'fold':
            action = valid_actions[0]['action']
            action_index = 0
            amount = 0
            self.folded = True
            self.times_folded += 1

        return action, amount, action_index

    def _print_episode(self, acc_reward, hand_info, round_state, won):
        if not self.episode_printout:
            return

        players = Players()
        hand_strength = ''
        if not self.folded:
            for player in hand_info:
                if player['uuid'] == self.uuid:
                    hand_strength = player['hand']['hand']['strength']

        print(
            "Episode: {}, won: {}, acc_reward: {:8.2f}, folded: {:2}, c_stack: {:8.2f}, total_bet: {:6.2f}, "
            "hand_ranking: {:3.2f}, risk_factor = {:4.2f}, hand_strength: {}, pref_allin: {:2}, players: {:1}".format(
                round_state['round_count'], won, acc_reward, self.folded, self.cashgame_stack,
                self.total_bet, self.hand_rank, self.risk_factor, hand_strength, self.preflop_allin,
                len(players.get_active_players(round_state['seats'], True, self.uuid))))

    def _use_preflop_statistic_network_decision(self, round_state, amount_to_play, hand_rank, play_blinds):
        limit = 0

        if round_state['street'] == 'preflop' and hand_rank < 0.3:
            if play_blinds:
                opponent_tightness = self.features.get_tightness_of_active_players(round_state['seats'], self.uuid)
                if opponent_tightness <= 0.6:  # loose opponent wants to collect blinds
                    limit = self.small_blind_amount * 2  # use network decision with low hand_rank, if only small amount
                    # is being played
            return amount_to_play <= limit

        return True

    def _update_agent_statistics(self, won, won_amount, total_bet):
        if self.plot_won_amount:
            self.won_amount_data.add_won_amount(won_amount)

        if self.plot_lost_amount:
            self.lost_amount_data.add_lost_amount(0 if won else total_bet)

        if self.plot_actions:
            self.action_data.complete_round(self.round_count)

    def _plot_agent_statistics(self):
        if self.rounds_to_play == self.round_count:
            if self.plot_cashgame_stack:
                plots = Plots.get_instance()
                plots.add_data_line_cashgame_stack(str(self), self.cashgame_stack_data.get_data())
                plots.plot_cash_game_stack_over_time(self.rounds_to_play)
            if self.plot_won_amount:
                plots = Plots.get_instance()
                plots.add_data_line_won_amount(str(self), self.won_amount_data.get_data())
                plots.plot_won_amount_over_time(self.rounds_to_play)
            if self.plot_lost_amount:
                plots = Plots.get_instance()
                plots.add_data_line_lost_amount(str(self), self.lost_amount_data.get_data())
                plots.plot_lost_amount_over_time(self.rounds_to_play)
            if self.plot_actions:
                plots = Plots.get_instance()
                plots.add_data_lines_actions(str(self), self.action_data.get_folds_data(),
                                             self.action_data.get_calls_data(), self.action_data.get_raises_data())
                plots.plot_actions_over_time(self.rounds_to_play)

    def _statistics_printouts(self):
        average_lost_amount = np.mean(self.lost_amount_history) if len(self.lost_amount_history) > 0 else 0
        average_won_amount = np.mean(self.won_amount_history) if len(self.won_amount_history) > 0 else 0
        total_lost_amount = np.sum(self.lost_amount_history)
        total_won_amount = np.sum(self.won_amount_history)
        average_round_actions_amount = np.mean(self.round_actions_length)

        print('\nGame Statistics for: {}'.format(self.__str__()))
        print('Reward model used: {}'.format(self.reward_model_used))
        print('Split network enabled: {} '.format(self.split_network))
        print('Play blinds: {} \n'.format(self.play_blinds))
        print('Times small called: {}'.format(self.times_small_called))
        print('Times big called: {}'.format(self.times_big_called))
        print('Times hand rank based raise: {}'.format(self.times_hr_raised))
        print('Times all in: {}'.format(self.times_all_in))
        print('Times preflop all in: {}'.format(self.preflop_allin_counter))
        print('Times high bet fold: {}'.format(self.high_bet_fold_counter))
        print('Times low hand rank fold: {}\n'.format(self.low_hand_rank_fold_counter))
        print('Average won amount: {}'.format(average_won_amount))
        print('Total won amount: {}'.format(total_won_amount))
        print('Length won amount: {}'.format(len(self.won_amount_history)))
        print('Average lost amount: {}'.format(average_lost_amount))
        print('Total lost amount: {}'.format(total_lost_amount))
        print('Length lost amount: {}\n'.format(len(self.lost_amount_history)))
        print('Average number of round actions: {}\n'.format(average_round_actions_amount))
        print('Percentage of games won: {}%'.format(self.times_won * 100 / self.rounds_to_play))
        rounds_played = self.rounds_to_play - self.times_folded
        print('Percentage of played games won (without folded): {}%'.format(
            (self.times_won * 100 / rounds_played) if rounds_played > 0 else 0))
        print('Percentage of folded games: {}%\n\n'.format(self.times_folded * 100 / self.rounds_to_play))

        return average_lost_amount, average_won_amount, total_lost_amount, total_won_amount, average_round_actions_amount

    def _send_statistics_mail(self, number_of_players, round_state, average_lost_amount, average_won_amount,
                              total_lost_amount, total_won_amount, average_round_actions_amount):
        gmail_user = 'pokerbot.competition@gmail.com'
        gmail_password = 'L3ts.P74y.P0k3r!'

        sent_from = gmail_user
        sent_to = ['vito.cudemo@students.fhnw.ch', 'michael.schaerz@students.fhnw.ch']
        subject = '{} Evaluation done'.format(self.player_name)

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sent_from
        msg['To'] = sent_to
        msg.set_content(
            'Statistics {} Bot Evaluation:\n\n'.format(self.player_name) +
            'Pretrained model loaded: {}\n'.format(self.agent.model_loaded) +
            'Reward model used: {} \n'.format(self.reward_model_used) +
            'Split network enabled: {} \n'.format(self.split_network) +
            'Play blinds: {} \n\n'.format(self.play_blinds) +
            'Times folded: {}\n'.format(self.times_folded) +
            'Times small called: {}\n'.format(self.times_small_called) +
            'Times big called: {}\n'.format(self.times_big_called) +
            'Times hand rank based raise: {}\n\n'.format(self.times_hr_raised) +
            'Times all in: {}\n'.format(self.times_all_in) +
            'Times preflop all in: {}\n'.format(self.preflop_allin_counter) +
            'Times high bet fold: {}\n'.format(self.high_bet_fold_counter) +
            'Times low hand rank fold: {}\n'.format(self.low_hand_rank_fold_counter) +
            'Average won amount: {}\n'.format(average_won_amount) +
            'Total won amount: {}\n'.format(total_won_amount) +
            'Average lost amount: {} \n'.format(average_lost_amount) +
            'Total lost amount: {} \n'.format(total_lost_amount) +
            'Average number of round actions: {}\n\n'.format(average_round_actions_amount) +
            'Rounds played: {}\n'.format(self.rounds_to_play) +
            'Number of Players: {}\n\n'.format(number_of_players) +
            'Percentage of played games won (without folded): {}%\n'.format(
                self.times_won * 100 / (self.rounds_to_play - self.times_folded)) +
            'Percentage of games won: {}%\n'.format(self.times_won * 100 / self.rounds_to_play) +
            'Percentage of folded games: {}%\n\n'.format(self.times_folded * 100 / self.rounds_to_play) +
            'Cashgame stack: {}\n\n'.format(self.cashgame_stack) +
            'Opponent statistics aggressivity: \n {} \n\n'.format(self.features.opponent_statistics_aggressivity) +
            'Opponent statistics tightness: \n {} \n\n'.format(self.features.opponent_statistics_tightness) +
            'Seats: \n {}'.format(round_state['seats'])
        )

        try:
            server_ssl = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server_ssl.ehlo()
            server_ssl.login(gmail_user, gmail_password)
            server_ssl.send_message(msg)
            server_ssl.close()
        except:
            print('Sent mail error occurred')

    def _send_start_mail(self, rounds_to_play):
        if not self.aicrowd_evaluation or not self.send_statistics_mail:
            return

        gmail_user = 'pokerbot.competition@gmail.com'
        gmail_password = 'L3ts.P74y.P0k3r!'

        sent_from = gmail_user
        sent_to = ['vito.cudemo@students.fhnw.ch', 'michael.schaerz@students.fhnw.ch']
        subject = 'AICrowd evaluation started'

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sent_from
        msg['To'] = sent_to
        msg.set_content(
            'Evaluation with {} rounds on AICrowd has started'.format(rounds_to_play)
        )
        try:
            server_ssl = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server_ssl.ehlo()
            server_ssl.login(gmail_user, gmail_password)
            server_ssl.send_message(msg)
            server_ssl.close()
        except:
            print('Sent mail error occurred')
