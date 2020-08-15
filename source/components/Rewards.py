import math


class Rewards:

    def get_reward_complex(self, done, won, won_amount, total_bet, amount_added, action_index,
                           minimum_call_amount, initial_stack, folded, street, risk_factor, hand_rank, tightness, blind_paid):

        # Reward System:
        # Reward cases used for setting up Reward System
        # Reward is added or subtracted in range (- 5 * initial_stack) to (+ 5 * initial_stack) to maintain balance
        # If folded at high risk and not high bet and counter player high tightness, give reward
        # If folded at high risk and not high bet and counter player low tightness, give offset reward
        # If folded at low risk and high bet, punish reward
        # If folded at low risk, punish reward
        # If folded at high risk + high bet, punish reward
        # If folded at high_bet, punish reward
        # If lost and used free call at low risk, give reward for nice try
        # If lost and raised at free call with low risk, punish reward
        # If lost and tightness threshold reached, punish reward
        # If lost and risk_factor threshold reached, punish reward
        # If lost and high_bet, punish reward
        # If won and used free call, give reward
        # If won and used free call to raise, give extra reward
        # If won and tightness threshold reached, give extra reward
        # If won and risk_factor threshold reached, give extra reward
        # If won and high_bet, give extra reward

        reward = -blind_paid

        # Hyperparameters
        free_call_threshold = 0
        hand_rank_threshold = 0.3  # Threshold based on odds calculations
        tightness_counter_threshold = 0.6
        risk_factor_threshold = 0.7  # Threshold based on odds calculations
        my_stack = initial_stack - total_bet

        low_risk = risk_factor < risk_factor_threshold
        high_risk = not low_risk
        low_tightness = tightness < tightness_counter_threshold
        high_tightness = not low_tightness
        no_bet = total_bet == 0
        high_bet = not no_bet
        all_in = initial_stack - total_bet == 0

        free_call = minimum_call_amount <= free_call_threshold

        # Case Folded
        # Up to 5 * initial_stack
        if folded and done:
            if high_risk and not high_bet:
                reward += my_stack  # Max: initial_stack
            elif high_bet:
                reward -= amount_added  # Max: initial_stack
            if high_tightness:
                reward += my_stack  # Max: initial_stack
            if street == 'preflop' and hand_rank < hand_rank_threshold:
                reward += my_stack  # Max: initial_stack
            elif street == 'preflop' and hand_rank >= hand_rank_threshold:
                reward -= my_stack  # Max: initial_stack

        # Up to 2 * initial_stack punishment
        elif folded and not done:
            if hand_rank >= hand_rank_threshold:
                reward -= my_stack * 2

        # Case Lost
        # Up to 5 * initial_stack punishment
        elif not won and action_index != 0:
            if action_index == 2 and free_call and high_risk:
                reward -= amount_added  # we should have called or folded
            if high_risk:
                reward -= amount_added
            if high_bet:
                reward -= amount_added
            if high_tightness:  # if others play tight and call/raise, we should fold
                reward -= amount_added
            if all_in:  # punish for to aggressive play
                reward -= amount_added

        # Won Case
        # Up to 5 * initial_stack + won_amount reward
        elif won:
            if action_index == 1 and free_call and low_risk:  # Called
                reward += amount_added  # Good call
            elif action_index == 2 and free_call and low_risk:  # Raised
                reward += amount_added * 2  # Even better - we raised when we could have free call
            elif action_index == 2 and low_risk:  # Raised
                reward += amount_added  # Even better - we raised at low risk
            if low_risk:
                reward += amount_added
            if high_bet:
                reward += amount_added
            if done:
                reward += amount_added + won_amount

        return reward


    def get_reward_simple(self, done, won, won_amount, amount_added, folded, blind_paid):

        # Case Folded
        if folded:
            return - amount_added - blind_paid

        # Case Lost

        elif not won:
           return - amount_added - blind_paid

        # Won Case
        if won:
            if not done:
                reward = amount_added
            if done:
                reward = amount_added + won_amount  # blind will be in won_amount
            return reward


    """
        if folded:
            # We know this round has been folded. Check for punishs, if we lost money or should have called
            if minimum_call_amount < free_call_threshold:
                reward -= offset

            if high_risk and low_bet:
                if high_tightness:  # Tight aggressive counter, tight player has good hand, so fold was right option
                    reward += minimum_call_amount + offset
                else:
                    reward = 0
            elif high_bet:
                if risk_factor < risk_factor_threshold:
                    reward -= amount_added
                else:
                    reward -= 2 * amount_added
            elif risk_factor < risk_factor_threshold:  # Small risk, small bet
                if tightness > tightness_counter_threshold:
                    reward = 0  # tight player has good hand, so fold was right option
                else:
                    reward -= amount_added
            else:
                reward -= amount_added
    """

    def get_reward_all_actions_sophisticated(self, done, won, won_amount, amount, initial_stack, total_bet, folded):

        if folded and total_bet < 100:
            return 0.15

        elif not done and won:
            return math.log2(amount + 10)  # case Call - Amount 0, and won in the end, give reward

        elif done and won:
            return math.log2(initial_stack + won_amount + amount)

        elif not won:
            return - \
                math.log2(total_bet)

        return 0

    def get_reward_won_or_small_loss_based(self, won, won_amount, initial_stack, total_bet, risk_level):
        small_loss = 0
        round_stack = initial_stack - total_bet  # how much money do I still have?

        if round_stack > initial_stack * risk_level:
            small_loss = 1

        reward = 1
        won_or_small_loss = won or small_loss

        if won_or_small_loss:
            if won:
                reward += initial_stack + won_amount
            if reward > 0:
                reward = math.log2(reward + (1 + round_stack + won_amount))

        return reward


    def get_reward_for_all_actions(self, done, won, won_amount, amount):

        if not done and won:
            reward = amount
            reward += 100  # case Call - Amount 0, and won in the end, give reward
            if reward > 0:
                return math.log2(reward)

        if done and won_amount > 0:
            return math.log2(won_amount + amount * 10)

        if not won and amount > 0:
            return 0  # Case Set amount but lost - punish

        # Case folded or not set any amount
        return 0.1


    def get_reward_multiplier_based(self, last_action, initial_stack, total_bet, folded,
                                    times_folded, times_called, times_raised, won, won_amount, round_state):
        # number_of_players = len(self._get_active_players(round_state['seats'], True))
        reward = 0
        round_stack = initial_stack - total_bet  # how much money do I still have?
        number_of_actions = times_folded + times_called + times_raised

        action_multiplier = 1
        if last_action == 'fold' and times_folded > 0:
            action_multiplier = number_of_actions / times_folded
        elif last_action == 'call' and times_called > 0:
            action_multiplier = number_of_actions / times_called
        elif times_raised > 0:
            action_multiplier = number_of_actions / times_raised

        if folded:
            reward = action_multiplier * (-1) * total_bet / initial_stack

            # player_factor = (number_of_players - 1) / (len(round_state['seats']) - 1)
            # self.reward = action_multiplier * player_factor * (round_stack / self.initial_stack)
        else:
            if won:
                # self.reward = action_multiplier * max(1, winner['stack'] / (number_of_players * 0.33 * self.initial_stack))
                # self.reward = action_multiplier * winner['stack'] / (len(round_state['seats']) * self.initial_stack)  # best so far ORIGINAL
                reward = action_multiplier * ((won_amount - total_bet) / (len(round_state[
                                                                                  'seats']) * initial_stack))  # best so far, UPDATED VCU for Rewards Class. ACHTUNG: WON_Amount wird bereits durch len(winners) dividiert.
                # self.reward = action_multiplier * max(1, winner['stack'] / (len(round_state['seats']) * 0.5 * self.initial_stack)) # bigger stack, but almost call-bot
            else:
                reward = action_multiplier * (-1) * total_bet / initial_stack

        return reward

        # Michael Version ORIGINAL
        # def _update_reward(self, winners, round_state):
        #     # number_of_players = len(self._get_active_players(round_state['seats'], True))
        #     won = 0
        #     round_stack = self.initial_stack - self.my_bet  # how much money do I still have?
        #     number_of_actions = self.times_folded + self.times_called + self.times_raised
        #
        #     action_multiplier = 1
        #     if self.last_action == 'fold' and self.times_folded > 0:
        #         action_multiplier = number_of_actions / self.times_folded
        #     elif self.last_action == 'call' and self.times_called > 0:
        #         action_multiplier = number_of_actions / self.times_called
        #     elif self.times_raised > 0:
        #         action_multiplier = number_of_actions / self.times_raised
        #
        #     if self.folded:
        #         self.reward = action_multiplier * round_stack / self.initial_stack
        #
        #         # player_factor = (number_of_players - 1) / (len(round_state['seats']) - 1)
        #         # self.reward = action_multiplier * player_factor * (round_stack / self.initial_stack)
        #     else:
        #         for winner in winners:
        #             if winner['uuid'] == self.uuid:
        #                 won = 1
        #                 # self.reward = action_multiplier * max(1, winner['stack'] / (number_of_players * 0.33 * self.initial_stack))
        #                 self.reward = action_multiplier * winner['stack'] / (
        #                             len(round_state['seats']) * self.initial_stack)  # best so far
        #                 # self.reward = action_multiplier * max(1, winner['stack'] / (len(round_state['seats']) * 0.5 * self.initial_stack)) # bigger stack, but almost call-bot
        #             else:
        #                 self.reward = action_multiplier * (-1) * self.my_bet / self.initial_stack
        #
        #     return won

        # V1 Reward
        # won = 0
        # round_stack = self.initial_stack - self.my_bet # how much money do I still have?
        #
        # if self.folded:
        #     self.reward = 2 * ((1.001 + round_stack)) #* (1.001 - self.win_rate))
        # else:
        #     self.reward = 2 * (-(1.001 - round_stack)) #* (1.001 - self.win_rate))**2  # higher the negative reward if not won
        #     for winner in winners:
        #         if winner['uuid'] == self.uuid:
        #             won = 1
        #             self.reward = (winner['stack'] - self.my_bet) #* self.win_rate
        #             self.cashgame_stack = winner['cashgame_stack'] + winner['stack']
        #             break
        #
        # if self.reward > 0:
        #     self.reward = math.log2(self.reward)
        #
        # return won
