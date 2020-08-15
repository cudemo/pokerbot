import numpy as np

suit_index_dict = {"s": 0, "c": 1, "h": 2, "d": 3}
reverse_suit_index = ("s", "c", "h", "d")
val_string = "AKQJT98765432"
hand_rankings = ("High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                 "Flush", "Full House", "Four of a Kind", "Straight Flush", "Royal Flush")
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in range(2, 10):
    suit_value_dict[str(num)] = num


class HoldemCalc:
    """
    Based on Kevin Tsengs Hold'em Calculator: https://github.com/ktseng/holdem_calc
    """

    def get_hand_probabilities(self, hole_cards, board):
        """
        Calculates the probability to get a certain hand ranking, based on the hole und community cards.
        :param hole_cards: Own players hole cards
        :param board: Community cards
        :return: List of Floats, containing the probability for each hand ranking
        """

        deck = self.__generate_deck(hole_cards, board)
        result_histogram = np.zeros(len(hand_rankings))
        self.__find_hand_probabilities(deck, hole_cards, board, result_histogram)

        number_of_probabilities = sum(result_histogram)
        for i in range(len(result_histogram)):
            result_histogram[i] = result_histogram[i] / number_of_probabilities

        return result_histogram

    def __find_hand_probabilities(self, deck, hole_cards, given_board, result_histograms):
        """
        Calculates the occurrences for each hand ranking.
        :param deck: All cards except the own hole and the community cards
        :param hole_cards: Own players hole cards
        :param given_board: Community cards
        :param result_histograms: List which will contain the probabilities for each hand ranking
        :return: Histogram of occurrences for each hand as List of Integers
        """

        # Run simulations
        for remaining_board in self.__generate_exhaustive_boards(deck, len(given_board)):
            # Generate a new board
            if given_board:
                board = given_board[:]
                board.extend(remaining_board)
            else:
                board = remaining_board

            # Find the best possible poker hand given the created board and the
            # hole cards and save them in the results data structures
            suit_histogram, histogram, max_suit = (self.__preprocess_board(board))
            result = self.__detect_hand(hole_cards, board, suit_histogram, histogram, max_suit)
            result_histograms[result[0]] += 1

    def __detect_hand(self, hole_cards, given_board, suit_histogram, full_histogram, max_suit):
        """
        Detects the hand ranking for e single combination of hole and community cards.
        :param hole_cards: Own players hole cards
        :param given_board: Community cards
        :param suit_histogram: Histogram for the colors in given_board
        :param full_histogram: Histogram for the values in given_board
        :param max_suit: Number of suits in given_board
        :return: List with information about hand strength. Only the first index, the hand ranking, will be used
        """

        # Determine if flush possible. If yes, four of a kind and full house are
        # impossible, so return royal, straight, or regular flush.
        if max_suit >= 3:
            flush_index = suit_histogram.index(max_suit)
            for hole_card in hole_cards:
                if hole_card.suit_index == flush_index:
                    max_suit += 1
            if max_suit >= 5:
                flat_board = list(given_board)
                flat_board.extend(hole_cards)
                suit_board = self.__generate_suit_board(flat_board, flush_index)
                result = self.__detect_straight_flush(suit_board)
                if result[0]:
                    return (8, result[1]) if result[1] != 14 else (9,)
                return 5, self.__get_high_cards(suit_board)

        # Add hole cards to histogram data structure and process it
        full_histogram = full_histogram[:]
        for hole_card in hole_cards:
            full_histogram[14 - hole_card.value] += 1
        histogram_board = self.__preprocess(full_histogram)

        # Find which card value shows up the most and second most times
        current_max, max_val, second_max, second_max_val = 0, 0, 0, 0
        for item in histogram_board:
            val, frequency = item[0], item[1]
            if frequency > current_max:
                second_max, second_max_val = current_max, max_val
                current_max, max_val = frequency, val
            elif frequency > second_max:
                second_max, second_max_val = frequency, val

        # Check to see if there is a four of a kind
        if current_max == 4:
            return 7, max_val, self.__detect_highest_quad_kicker(histogram_board)
        # Check to see if there is a full house
        if current_max == 3 and second_max >= 2:
            return 6, max_val, second_max_val
        # Check to see if there is a straight
        if len(histogram_board) >= 5:
            result = self.__detect_straight(histogram_board)
            if result[0]:
                return 4, result[1]
        # Check to see if there is a three of a kind
        if current_max == 3:
            return 3, max_val, self.__detect_three_of_a_kind_kickers(histogram_board)
        if current_max == 2:
            # Check to see if there is a two pair
            if second_max == 2:
                return 2, max_val, second_max_val, self.__detect_highest_kicker(
                    histogram_board)
            # Return pair
            else:
                return 1, max_val, self.__detect_pair_kickers(histogram_board)
        # Check for high cards
        return 0, self.__get_high_cards(histogram_board)

    def __detect_straight_flush(self, suit_board):
        """
        Checks if there is a Straight Flush
        :param suit_board: Card values of the suited board
        :return: Tuple: (Is there a straight flush?, high card)
        """

        contiguous_length, fail_index = 1, len(suit_board) - 5
        # Won't overflow list because we fail fast and check ahead
        for index, elem in enumerate(suit_board):
            current_val, next_val = elem, suit_board[index + 1]
            if next_val == current_val - 1:
                contiguous_length += 1
                if contiguous_length == 5:
                    return True, current_val + 3
            else:
                # Fail fast if straight not possible
                if index >= fail_index:
                    if (index == fail_index and next_val == 5 and
                            suit_board[0] == 14):
                        return True, 5
                    break
                contiguous_length = 1
        return False,

    def __detect_highest_quad_kicker(self, histogram_board):
        """
        Returns the highest Kicker available for a four of a kind,
        :param histogram_board: List of Integer Tuples (card value, number of occurrences)
        :return: Card value of the Kicker
        """

        for elem in histogram_board:
            if elem[1] < 4:
                return elem[0]

    def __detect_straight(self, histogram_board):
        """
        Checks if a Straight is present
        :param histogram_board: List of Integer Tuples (card value, number of occurrences)
        :return: Tuple: (Is there a straight?, high card)
        """

        contiguous_length, fail_index = 1, len(histogram_board) - 5
        # Won't overflow list because we fail fast and check ahead
        for index, elem in enumerate(histogram_board):
            current_val, next_val = elem[0], histogram_board[index + 1][0]
            if next_val == current_val - 1:
                contiguous_length += 1
                if contiguous_length == 5:
                    return True, current_val + 3
            else:
                # Fail fast if straight not possible
                if index >= fail_index:
                    if (index == fail_index and next_val == 5 and
                            histogram_board[0][0] == 14):
                        return True, 5
                    break
                contiguous_length = 1
        return False,

    def __detect_three_of_a_kind_kickers(self, histogram_board):
        """
        Returns tuple of the two highest kickers that result from the three of a kind
        :param histogram_board: List of Integer Tuples (card value, number of occurrences)
        :return: Highest kickers
        """

        kicker1 = -1
        for elem in histogram_board:
            if elem[1] != 3:
                if kicker1 == -1:
                    kicker1 = elem[0]
                else:
                    return kicker1, elem[0]

    def __detect_highest_kicker(self, histogram_board):
        """
        Gets the highest kicker available
        :param histogram_board: List of Integer Tuples (card value, number of occurrences)
        :return: Highest available Kicker
        """

        for elem in histogram_board:
            if elem[1] == 1:
                return elem[0]

    def __detect_pair_kickers(self, histogram_board):
        """
        Get the Kickers for a pair
        :param histogram_board: List of Integer Tuples (card value, number of occurrences)
        :return: Tuple: (kicker1, kicker2, kicker3)
        """

        kicker1, kicker2 = -1, -1
        for elem in histogram_board:
            if elem[1] != 2:
                if kicker1 == -1:
                    kicker1 = elem[0]
                elif kicker2 == -1:
                    kicker2 = elem[0]
                else:
                    return kicker1, kicker2, elem[0]

    def __get_high_cards(self, histogram_board):
        """
        Gets the five highest kickers.
        :param histogram_board: Sorted board
        :return: List of the five highest cards in the given board
        """

        return histogram_board[:5]

    def __preprocess_board(self, flat_board):
        """
        Takes an iterable sequence and returns two items in a tuple:
        1: 4-long list showing how often each card suit appears in the sequence
        2: 13-long list showing how often each card value appears in the sequence
        :param flat_board: Community cards
        :return: Suit histogram, Value histogram, highest number of suits in flat_board
        """

        suit_histogram, histogram = [0] * 4, [0] * 13
        # Reversing the order in histogram so in the future, we can traverse
        # starting from index 0
        for card in flat_board:
            histogram[14 - card.value] += 1
            suit_histogram[card.suit_index] += 1
        return suit_histogram, histogram, max(suit_histogram)

    def __generate_deck(self, hole_cards, board):
        """
        Returns deck of cards with all hole cards and board cards removed
        :param hole_cards: Own players hole cards
        :param board: Community cards
        :return: All available cards, except the hole and community cards
        """

        deck = []
        for suit in reverse_suit_index:
            for value in val_string:
                deck.append(Card(value + suit))

        taken_cards = []
        for card in hole_cards:
            taken_cards.append(card)
        if board and len(board) > 0:
            taken_cards.extend(board)
        for taken_card in taken_cards:
            deck.remove(taken_card)
        return tuple(deck)

    def __generate_exhaustive_boards(self, deck, board_length):
        """
        Generate all possible boards.
        :param deck: Deck containing all cards which have not yet been used
        :param board_length: Number of community cards which have already been revealed
        :return: All possible boards, based on the available cards in the deck
        """

        import itertools
        return itertools.combinations(deck, 5 - board_length)

    def __preprocess(self, histogram):
        """
        Returns a list of two tuples of the form: (value of card, frequency of card)
        :param histogram: Histogram of card values
        :return: List of tuples (value of card, frequency of card)
        """

        return [(14 - index, frequency) for index, frequency in
                enumerate(histogram) if frequency]

    def __generate_suit_board(self, flat_board, flush_index):
        """
        Returns a board of cards all with suit = flush_index
        :param flat_board:
        :param flush_index: Color code for the Flush
        :return: Board with given Color
        """

        histogram = [card.value for card in flat_board
                     if card.suit_index == flush_index]
        histogram.sort(reverse=True)
        return histogram


class Card:

    def __init__(self, card_string):
        """
        Creates a new card.
        :param card_string: String in the format: "As", "Tc", "6d"
        """

        value, self.suit = card_string[0], card_string[1]
        self.value = suit_value_dict[value]
        self.suit_index = suit_index_dict[self.suit]

    def __str__(self):
        return val_string[14 - self.value] + self.suit

    def __repr__(self):
        return val_string[14 - self.value] + self.suit

    def __eq__(self, other):
        if self is None:
            return other is None
        elif other is None:
            return False
        return self.value == other.value and self.suit == other.suit
