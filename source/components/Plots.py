import configparser
import datetime
from math import ceil

import matplotlib.pyplot as plt


class Plots:
    __instance = None  # Singleton instance of the Plots class

    def __init__(self):
        """
        Avoid creating a new instance of the class if the instance has already been created
        """
        if Plots.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Plots.__instance = self
            self.__data_cashgame_stack = {}
            self.__data_won_amount = {}
            self.__data_lost_amount = {}
            self.__data_actions = {}

    @staticmethod
    def get_instance():
        """
        Method to be used when accessing the Plots class
        :return: Singleton instance of class Plots
        """
        if Plots.__instance is None:
            Plots()
        return Plots.__instance

    def register_agent_cashgame_stack(self, agent_name):
        """
        Registers a new agent to plot Cashgame Stack development
        :param agent_name: Name of the agent which will be used on the plot
        :return: Nothing. Creates the required dictionary for the Cashgame Stack plot
        """
        self.__data_cashgame_stack[agent_name] = {
            'ready': False,
            'data_lines': []
        }

    def register_agent_won_amount(self, agent_name):
        """
        Registers a new agent to plot average won amount
        :param agent_name: Name of the agent which will be used on the plot
        :return: Nothing. Creates the required dictionary for the average won amount plot
        """
        self.__data_won_amount[agent_name] = {
            'ready': False,
            'data_lines': []
        }

    def register_agent_lost_amount(self, agent_name):
        """
        Registers a new agent to plot average lost amount
        :param agent_name: Name of the agent which will be used on the plot
        :return: Nothing. Creates the required dictionary for the average lost amount plot
        """
        self.__data_lost_amount[agent_name] = {
            'ready': False,
            'data_lines': []
        }

    def register_agent_actions(self, agent_name):
        """
        Registers a new agent to plot chosen actions
        :param agent_name: Name of the agent which will be used on the plot
        :return: Nothing. Creates the required dictionary for the actions plot
        """
        self.__data_actions[agent_name] = {
            'ready': False,
            'data_folds': [],
            'data_calls': [],
            'data_raises': []
        }

    def add_data_line_cashgame_stack(self, key, data):
        """
        Adds the definitive Cashgame Stack data of an agent to the plot and marks the agent as ready to plot
        :param key: Agent, which delivers the data
        :param data: Cashgame Stack data to plot
        :return: Nothing. Sets the data to plot
        """
        self.__data_cashgame_stack[key]['ready'] = True
        self.__data_cashgame_stack[key]['data_lines'] = data

    def add_data_line_won_amount(self, key, data):
        """
        Adds the definitive average won amount data of an agent to the plot and marks the agent as ready to plot
        :param key: Agent, which delivers the data
        :param data: Average won amount data to plot
        :return: Nothing. Sets the data to plot
        """
        self.__data_won_amount[key]['ready'] = True
        self.__data_won_amount[key]['data_lines'] = data

    def add_data_line_lost_amount(self, key, data):
        """
        Adds the definitive average lost amount data of an agent to the plot and marks the agent as ready to plot
        :param key: Agent, which delivers the data
        :param data: Average lost amount data to plot
        :return: Nothing. Sets the data to plot
        """
        self.__data_lost_amount[key]['ready'] = True
        self.__data_lost_amount[key]['data_lines'] = data

    def add_data_lines_actions(self, key, data_folds, data_calls, data_raises):
        """
        Adds the definitive actions data of an agent to the plot and marks the agent as ready to plot
        :param key: Agent, which delivers the data
        :param data_folds: Folds data to plot
        :param data_calls: Calls data to plot
        :param data_raises: Raises data to plot
        :return: Nothing. Sets the actions data to plot
        """
        self.__data_actions[key]['ready'] = True
        self.__data_actions[key]['data_folds'] = data_folds
        self.__data_actions[key]['data_calls'] = data_calls
        self.__data_actions[key]['data_raises'] = data_raises

    def plot_cash_game_stack_over_time(self, total_rounds):
        """
        Plots the Cashgame Stack data for all registered agents or cancels, if the agents are not ready yet
        :param total_rounds: Number of rounds that have been played
        :return: Nothing. Creates the plot images
        """
        if not self._all_agents_ready_to_plot_cashgame_stack():
            return

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        step_size_cashgame_stack = config['plot_settings'].getint('step_size_cashgame_stack')

        x = range(0, total_rounds, step_size_cashgame_stack)

        fig = plt.figure(figsize=(10, 7.5))
        for key in self.__data_cashgame_stack.keys():
            plt.plot(x, self.__data_cashgame_stack[key]['data_lines'], label=key)
        fig.suptitle('Cashgame stack development for ' + str(step_size_cashgame_stack) + ' played hands')
        plt.xlabel('Hands played')
        plt.ylabel('Cashgame stack')
        plt.legend()

        image_path = config['plot_settings']['file_path_cashgame_stack']
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        image_path = image_path.format(now)
        fig.savefig(image_path)

    def plot_won_amount_over_time(self, total_rounds):
        """
        Plots the average won amount data for all registered agents or cancels, if the agents are not ready yet
        :param total_rounds: Number of rounds that have been played
        :return: Nothing. Creates the plot images
        """
        if not self._all_agents_ready_to_plot_won_amount():
            return

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        # ddqn_replace_interval = config['model_settings'].getint('ddqn_replace_network_interval')
        step_size_won_amount = config['plot_settings'].getint('step_size_won_amount')

        x = range(0, total_rounds, step_size_won_amount)
        # x_replace = range(ddqn_replace_interval, total_rounds, ddqn_replace_interval)

        fig = plt.figure(figsize=(10, 7.5))
        for key in self.__data_won_amount.keys():
            plt.plot(x, self.__data_won_amount[key]['data_lines'], label=key)
        # for x in x_replace:
        #     plt.axvline(x, c='y')
        fig.suptitle('Average of chips won per ' + str(step_size_won_amount) + ' played hands')
        plt.xlabel('Hands played')
        plt.ylabel('Average chips won')
        plt.legend()

        image_path = config['plot_settings']['file_path_won_amount']
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        image_path = image_path.format(now)
        fig.savefig(image_path)

    def plot_lost_amount_over_time(self, total_rounds):
        """
        Plots the average lost amount data for all registered agents or cancels, if the agents are not ready yet
        :param total_rounds: Number of rounds that have been played
        :return: Nothing. Creates the plot images
        """
        if not self._all_agents_ready_to_plot_lost_amount():
            return

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        step_size_lost_amount = config['plot_settings'].getint('step_size_lost_amount')

        x = range(0, total_rounds, step_size_lost_amount)

        fig = plt.figure(figsize=(10, 7.5))
        for key in self.__data_lost_amount.keys():
            plt.plot(x, self.__data_lost_amount[key]['data_lines'], label=key)
        fig.suptitle('Average of chips lost per ' + str(step_size_lost_amount) + ' played hands')
        plt.xlabel('Hands played')
        plt.ylabel('Average chips lost')
        plt.legend()

        image_path = config['plot_settings']['file_path_lost_amount']
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        image_path = image_path.format(now)
        fig.savefig(image_path)

    def plot_actions_over_time(self, total_rounds):
        """
        Plots the action data for all registered agents or cancels, if the agents are not ready yet
        :param total_rounds: Number of rounds that have been played
        :return: Nothing. Creates the plot images
        """
        if not self._all_agents_ready_to_plot_actions():
            return

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        step_size_actions = config['plot_settings'].getint('step_size_actions')

        x = range(0, total_rounds, step_size_actions)

        for key in self.__data_actions.keys():
            fig = plt.figure(figsize=(10, 7.5))
            plt.plot(x, self.__data_actions[key]['data_folds'], label='Folds')
            plt.plot(x, self.__data_actions[key]['data_calls'], label='Calls')
            plt.plot(x, self.__data_actions[key]['data_raises'], label='Raises')

            fig.suptitle('Sum of actions per ' + str(step_size_actions) + ' played hands for ' + str(key))
            plt.xlabel('Hands played')
            plt.ylabel('Sum of action')
            plt.legend()

            image_path = config['plot_settings']['file_path_actions']
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            image_path = image_path.format(key, now)
            fig.savefig(image_path)

    def _all_agents_ready_to_plot_cashgame_stack(self):
        """
        Check whether all agents have the required data to create the Cashgame Stack plot
        :return: True if all agents are ready to plot. False if at least one agent is not yet ready to plot
        """
        if len(self.__data_cashgame_stack) == 0:
            return False

        for key in self.__data_cashgame_stack.keys():
            if not self.__data_cashgame_stack[key]['ready']:
                return False
        return True

    def _all_agents_ready_to_plot_won_amount(self):
        """
        Check whether all agents have the required data to create the won amount plot
        :return: True if all agents are ready to plot. False if at least one agent is not yet ready to plot
        """
        if len(self.__data_won_amount) == 0:
            return False

        for key in self.__data_won_amount.keys():
            if not self.__data_won_amount[key]['ready']:
                return False
        return True

    def _all_agents_ready_to_plot_lost_amount(self):
        """
        Check whether all agents have the required data to create the lost amount plot
        :return: True if all agents are ready to plot. False if at least one agent is not yet ready to plot
        """
        if len(self.__data_lost_amount) == 0:
            return False

        for key in self.__data_lost_amount.keys():
            if not self.__data_lost_amount[key]['ready']:
                return False
        return True

    def _all_agents_ready_to_plot_actions(self):
        """
        Check whether all agents have the required data to create the actions plot
        :return: True if all agents are ready to plot. False if at least one agent is not yet ready to plot
        """
        if len(self.__data_actions) == 0:
            return False

        for key in self.__data_actions.keys():
            if not self.__data_actions[key]['ready']:
                return False
        return True


class CashgameStackData:

    def __init__(self):
        self.__data = []
        self.__number_of_data_points = 0
        self.__total_rounds = 0

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        self.__step_size_cashgame_stack = config['plot_settings'].getint('step_size_won_amount')

    def get_data(self):
        """
        Returns the collected data line
        :return: Cashgame Stack data
        """
        return self.__data

    def set_total_rounds(self, total_rounds):
        """
        Sets the number of rounds that have to be played
        :param total_rounds: Total number of rounds to play
        """
        self.__total_rounds = total_rounds
        self.__number_of_data_points = ceil(total_rounds / self.__step_size_cashgame_stack)

    def add_cashgame_stack(self, cashgame_stack, round_number):
        """
        Adds a new entry to the Cashgame Stack data line
        :param cashgame_stack: Data point on the Cashgame Stack data line
        :param round_number: Current round number
        """
        if round_number % self.__step_size_cashgame_stack == 0:
            self.__data.append(cashgame_stack)


class WonAmountData:

    def __init__(self):
        self.__data = []
        self.__temp_data = []
        self.__number_of_data_points = 0
        self.__total_rounds = 0

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        self.__step_size_won_amount = config['plot_settings'].getint('step_size_won_amount')

    def get_data(self):
        """
        Returns the collected data line
        :return: Won amount data
        """
        return self.__data

    def set_total_rounds(self, total_rounds):
        """
        Sets the number of rounds that have to be played
        :param total_rounds: Total number of rounds to play
        """
        self.__total_rounds = total_rounds
        self.__number_of_data_points = ceil(total_rounds / self.__step_size_won_amount)

    def add_won_amount(self, won_amount):
        """
        Adds a new entry to the won amount data line
        :param won_amount: Data point on the won amount data line
        """
        self.__temp_data.append(won_amount)

        if len(self.__temp_data) == self.__step_size_won_amount:
            self.__data.append(sum(self.__temp_data) / len(self.__temp_data))  # Average
            self.__temp_data = []


class LostAmountData:

    def __init__(self):
        self.__data = []
        self.__temp_data = []
        self.__number_of_data_points = 0
        self.__total_rounds = 0

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        self.__step_size_lost_amount = config['plot_settings'].getint('step_size_lost_amount')

    def get_data(self):
        """
        Returns the collected data line
        :return: Lost amount data
        """
        return self.__data

    def set_total_rounds(self, total_rounds):
        """
        Sets the number of rounds that have to be played
        :param total_rounds: Total number of rounds to play
        """
        self.__total_rounds = total_rounds
        self.__number_of_data_points = ceil(total_rounds / self.__step_size_lost_amount)

    def add_lost_amount(self, lost_amount):
        """
        Adds a new entry to the lost amount data line
        :param lost_amount: Data point on the lost amount data line
        """
        self.__temp_data.append(lost_amount)

        if len(self.__temp_data) == self.__step_size_lost_amount:
            self.__data.append(sum(self.__temp_data) / len(self.__temp_data))
            self.__temp_data = []


class ActionData:

    def __init__(self):
        self.__data_folds = []
        self.__data_calls = []
        self.__data_raises = []
        self.__temp_count_folds = 0
        self.__temp_count_calls = 0
        self.__temp_count_raises = 0
        self.__number_of_data_points = 0
        self.__total_rounds = 0

        config = configparser.ConfigParser()
        config.read('configuration/agent_config.ini')
        self.__step_size_actions = config['plot_settings'].getint('step_size_actions')

    def set_total_rounds(self, total_rounds):
        """
        Sets the number of rounds that have to be played
        :param total_rounds: Total number of rounds to play
        """
        self.__total_rounds = total_rounds
        self.__number_of_data_points = ceil(total_rounds / self.__step_size_actions)

    def get_folds_data(self):
        """
        Returns the collected folds data line
        :return: Folds data
        """
        return self.__data_folds

    def get_calls_data(self):
        """
        Returns the collected calls data line
        :return: Calls data
        """
        return self.__data_calls

    def get_raises_data(self):
        """
        Returns the collected folds data line
        :return: Raises data
        """
        return self.__data_raises

    def add_action(self, action):
        """
        Increases the counter for the given action
        :param action: Name of the action executed by an agent
        """
        if action == 'fold':
            self.__temp_count_folds += 1
        elif action == 'call':
            self.__temp_count_calls += 1
        elif action == 'raise':
            self.__temp_count_raises += 1

    def complete_round(self, round_number):
        """
        Adds the temporary data to the corresponding data line if the step size for the actions is reached
        :param round_number: Current round number
        """
        if round_number % self.__step_size_actions == 0:
            self.__data_folds.append(self.__temp_count_folds)
            self.__data_calls.append(self.__temp_count_calls)
            self.__data_raises.append(self.__temp_count_raises)
            self.__temp_count_folds = 0
            self.__temp_count_calls = 0
            self.__temp_count_raises = 0
