import numpy as np


# Use separate lists for memory with zero-initialized matrices to speed up learning
INDICES = np.arange(32)

class MemoryBuffer(object):

    def __init__(self, max_size, number_of_parameters, with_per=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, number_of_parameters), np.int8)
        self.new_state_memory = np.zeros((self.mem_size, number_of_parameters), np.int8)
        self.action_memory = np.zeros((self.mem_size), np.int8)
        self.reward_memory = np.zeros(self.mem_size, np.float32)
        self.terminal_memory = np.zeros(self.mem_size, np.int8)
        self.priority_memory = np.zeros(self.mem_size, np.float32)
        self.last_index = 0
        self.with_per = with_per

    def store_state(self, state, action, reward, new_state, done):
        """
        Will be used by Models to remember a state transition.
        :param state: the state before the transition to save in state_memory
        :param action: the action taken at that state to save in action_memory
        :param reward: the reward taken from that action at that state to save in reward_memory
        :param new_state: the new_state we got from choosing action at that state to save in next_state_memory
        :param done: done flag, if round is finished to save in terminal_memory. will be saved inverted.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.with_per:
            if max(self.priority_memory) == 0:
                self.priority_memory[index] = 1  # initialize with max priority so that it is being selected
            else:
                self.priority_memory[index] = max(self.priority_memory)

        self.mem_cntr += 1
        self.last_index = index

    def get_sample_batch(self, batch_size, priority_scale=1.0):
        """
        Generates a random and, if enabled, prioritized as well as importance_labeled list of elements to use for replay
        :param batch_size: number of elements that should be sampled
        :param priority_scale: If prioritized experience replay is enabled, a priority_scale can be set. The scale will
        be used as Beta parameter based on Prioritized Experience Replay Paper, Algorithm1 Listing, 9
        :return: list of sampled batch with size batch_size
        """
        max_memory = min(self.mem_cntr, self.mem_size)

        if self.with_per:
            probabilities = self.get_probabilities(priority_scale)
            batch_indices = np.random.choice(max_memory, batch_size, p=probabilities)  # set weight for being picked
            states = self.state_memory[batch_indices]
            actions = self.action_memory[batch_indices]
            rewards = self.reward_memory[batch_indices]
            new_states = self.new_state_memory[batch_indices]
            terminals = self.terminal_memory[batch_indices]
            importances = self.get_importance(probabilities[batch_indices], batch_size)
            return states, actions, rewards, new_states, terminals, importances, batch_indices
        else:
            batch_indices = np.random.choice(max_memory, batch_size)  # get random indices
            states = self.state_memory[batch_indices]
            actions = self.action_memory[batch_indices]
            rewards = self.reward_memory[batch_indices]
            new_states = self.new_state_memory[batch_indices]
            terminals = self.terminal_memory[batch_indices]
            return states, actions, rewards, new_states, terminals

    def get_probabilities(self, priority_scale):
        """
       Sample transition based on Prioritized Experience Replay Paper, Algorithm1 Listing, 9
       :param priority_scale: scale priority value Beta based on Prioritized Experience Replay Paper
       :return sample_probabilities: list containing sampled probability scale for each index in dataset
       """
        max_memory = min(self.mem_cntr, self.mem_size)
        scaled_priorities = np.array(self.priority_memory[0:max_memory]) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities, batch_size):
        """
        Compute importance-sampling weight w based on Prioritized Experience Replay Paper, Algorithm1 Listing, 10
        Importance is the criterion by which the network knows the amount it can learn from a specific transition.
        :param probabilities: computed probabilities for dataset being used
        :param batch_size: the batch_size being used to normalize the importance variances
        :return importance_normalized: list with computed importances for each index in batch
        """
        importance = 1 / batch_size * 1 / probabilities
        importance_normalized = importance / max(importance)  # for stability reasons normalize by max weight w
        return importance_normalized

    def set_priorities(self, indices, td_errors, offset=0.1):
        """
        Use td_error now to compute priority for prioritized experience replay
        :param indices: indices of affected batch
        :param td_errors: computed temporal difference error
        :param offset: we do not want priority 0, therefore offset with default 0.1
        """
        for i, e in zip(indices, td_errors):
            self.priority_memory[i] = abs(e) + offset
