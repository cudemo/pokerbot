import unittest

from components.Features import Features


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.features = Features()
        self.uuid = 'aaaaaaaaaaaaaaaaaaaaaa'

    def test_get_street_one_hot_preflop(self):
        # Act
        street_one_hot = self.features.get_street_one_hot('preflop')

        # Assert
        self.assertEqual(street_one_hot.tolist(), [1, 0, 0, 0])

    def test_get_street_one_hot_flop(self):
        # Act
        street_one_hot = self.features.get_street_one_hot('flop')

        # Assert
        self.assertEqual(street_one_hot.tolist(), [0, 1, 0, 0])

    def test_get_street_one_hot_turn(self):
        # Act
        street_one_hot = self.features.get_street_one_hot('turn')

        # Assert
        self.assertEqual(street_one_hot.tolist(), [0, 0, 1, 0])

    def test_get_street_one_hot_river(self):
        # Act
        street_one_hot = self.features.get_street_one_hot('river')

        # Assert
        self.assertEqual(street_one_hot.tolist(), [0, 0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_zero_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_zero_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_zero_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_zero_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_position_zero_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_zero_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_one_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_two_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_three_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_four_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_six_players_small_blind_index_five_player_index_five(self):
        # Arrange
        round_state = {
            'small_blind_pos': 5,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': 'ffffffffffffffffffffff', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_zero_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_zero_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_zero_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_five_players_small_blind_index_zero_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_position_zero_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_one_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_one_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_one_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_one_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_five_players_small_blind_position_one_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_two_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_two_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_two_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_two_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_position_two_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_five_players_small_blind_index_three_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_five_players_small_blind_index_three_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_three_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_three_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_position_three_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_four_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_five_players_small_blind_index_four_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_five_players_small_blind_index_four_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_index_four_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_five_players_small_blind_position_four_player_index_four(self):
        # Arrange
        round_state = {
            'small_blind_pos': 4,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': 'eeeeeeeeeeeeeeeeeeeeee', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_zero_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_zero_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_zero_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_four_players_small_blind_index_zero_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_four_players_small_blind_index_one_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_four_players_small_blind_index_one_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_one_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_one_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_four_players_small_blind_index_two_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_four_players_small_blind_index_two_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_four_players_small_blind_index_two_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_two_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_three_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_four_players_small_blind_index_three_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_four_players_small_blind_index_three_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_four_players_small_blind_index_three_player_index_three(self):
        # Arrange
        round_state = {
            'small_blind_pos': 3,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_three_players_small_blind_index_zero_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_three_players_small_blind_index_zero_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_three_players_small_blind_index_zero_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_three_players_small_blind_index_one_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_three_players_small_blind_index_one_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_three_players_small_blind_index_one_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_three_players_small_blind_index_two_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 1, 0])

    def test_get_position_one_hot_three_players_small_blind_index_two_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_three_players_small_blind_index_two_player_index_two(self):
        # Arrange
        round_state = {
            'small_blind_pos': 2,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_two_players_small_blind_index_zero_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_position_one_hot_two_players_small_blind_index_zero_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 0,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_two_players_small_blind_index_one_player_index_zero(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [0, 0, 1])

    def test_get_position_one_hot_two_players_small_blind_index_one_player_index_one(self):
        # Arrange
        round_state = {
            'small_blind_pos': 1,
            'seats': [
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': self.uuid, 'state': 'participating'}
            ]
        }

        # Act
        position_one_hot = self.features.get_position_one_hot(round_state, self.uuid)

        # Assert
        self.assertEqual(position_one_hot.tolist(), [1, 0, 0])

    def test_get_hole_card_rank_all_combinations_available(self):
        # Arrange
        card_values = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        card_colors = ['c', 'd', 'h', 's']

        for value_1 in card_values:
            for value_2 in card_values:
                for color_1 in card_colors:
                    for color_2 in card_colors:
                        card_1 = value_1 + color_1
                        card_2 = value_2 + color_2

                        if card_1 == card_2:
                            continue

                        # Act
                        hole_card_rank = self.features._get_hole_card_rank([card_1, card_2])

                        # Assert
                        self.assertTrue(hole_card_rank >= 0)

    def test_get_hole_card_rank_strongest(self):
        # Act
        hole_card_rank = self.features._get_hole_card_rank(['As', 'Ad'])

        # Assert
        self.assertEqual(hole_card_rank, 1)

    def test_get_hole_card_rank_weakest(self):
        # Act
        hole_card_rank = self.features._get_hole_card_rank(['2h', '7c'])

        # Assert
        self.assertEqual(hole_card_rank, 0)

    def test_get_tightness_of_active_players_all_players_active(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0
            },
            'cccccccccccccccccccccc': {
                'tightness': 0
            },
            'dddddddddddddddddddddd': {
                'tightness': 0.9
            }
        }

        seats = [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'participating'},
        ]

        # Act
        tightness = self.features.get_tightness_of_active_players(seats, self.uuid)

        # Assert
        self.assertEqual(tightness, 0.3)

    def test_get_tightness_of_active_players_not_all_players_active(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.5
            },
            'cccccccccccccccccccccc': {
                'tightness': 0.2
            },
            'dddddddddddddddddddddd': {
                'tightness': 0.9
            }
        }

        seats = [
                {'uuid': self.uuid, 'state': 'participating'},
                {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
                {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'},
                {'uuid': 'dddddddddddddddddddddd', 'state': 'folded'},
        ]

        # Act
        tightness = self.features.get_tightness_of_active_players(seats, self.uuid)

        # Assert
        self.assertEqual(tightness, 0.35)

    def test_get_tightness_one_hot_very_tight_lower_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.81
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [1, 0, 0, 0, 0])

    def test_get_tightness_one_hot_very_tight(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.9
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [1, 0, 0, 0, 0])

    def test_get_tightness_one_hot_very_tight_upper_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 1
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [1, 0, 0, 0, 0])

    def test_get_tightness_one_hot_tight_lower_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.61
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 1, 0, 0, 0])

    def test_get_tightness_one_hot_tight(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.7
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 1, 0, 0, 0])

    def test_get_tightness_one_hot_tight_upper_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.8
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 1, 0, 0, 0])

    def test_get_tightness_one_hot_balanced_lower_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.41
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 1, 0, 0])

    def test_get_tightness_one_hot_balanced(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.5
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 1, 0, 0])

    def test_get_tightness_one_hot_balanced_upper_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.6
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 1, 0, 0])

    def test_get_tightness_one_hot_loose_lower_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.21
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 1, 0])

    def test_get_tightness_one_hot_loose(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.3
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 1, 0])

    def test_get_tightness_one_hot_loose_upper_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.4
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 1, 0])

    def test_get_tightness_one_hot_very_loose_lower_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 0, 1])

    def test_get_tightness_one_hot_very_loose(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.1
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 0, 1])

    def test_get_tightness_one_hot_very_loose_upper_limit(self):
        # Arrange
        self.features.opponent_statistics_tightness = {
            'bbbbbbbbbbbbbbbbbbbbbb': {
                'tightness': 0.2
            }
        }

        seats = [
            {'uuid': self.uuid, 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
        ]

        # Act
        tightness_one_hot = self.features.get_tightness_one_hot(seats, self.uuid)

        # Assert
        self.assertEqual(tightness_one_hot.tolist(), [0, 0, 0, 0, 1])

    def test_get_cards_rank_royal_flush(self):
        # Arrange
        hole_card = ['Ah', 'Kh']
        round_state = {
            'community_card': ['Qh', 'Jh', 'Th']
        }

        # Act
        hand_rank_pair = self.features._get_cards_rank(hole_card, round_state, False)

        # Assert
        self.assertTrue(hand_rank_pair > 0.99)

    def test_get_cards_rank_pair(self):
        # Arrange
        hole_card = ['Ad', '4s']
        round_state = {
            'community_card': ['Ah', 'Kc', '8d']
        }

        # Act
        hand_rank_pair = self.features._get_cards_rank(hole_card, round_state, False)

        # Assert
        self.assertTrue(hand_rank_pair > 0.54)
        self.assertTrue(hand_rank_pair < 0.55)

    def test_get_cards_rank_worst(self):
        # Arrange
        hole_card = ['2d', '7s']
        round_state = {
            'community_card': ['5h', '4c', '3d']
        }

        # Act
        hand_rank_pair = self.features._get_cards_rank(hole_card, round_state, False)

        # Assert
        self.assertTrue(hand_rank_pair < 0.01)

    def test_get_pot_size(self):
        # Act
        pot_size = self.features.get_pot_size(300, 6, 200)

        # Assert
        self.assertEqual(pot_size, 0.25)

    def test_get_pot_size_lowest(self):
        # Act
        pot_size_one_hot = self.features.get_pot_size_one_hot(3, 6, 200)

        # Assert
        self.assertEqual(pot_size_one_hot.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_pot_size_highest(self):
        # Act
        pot_size_one_hot = self.features.get_pot_size_one_hot(1100, 6, 200)

        # Assert
        self.assertEqual(pot_size_one_hot.tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_get_pot_odds_free(self):
        # Act
        pot_odds = self.features.get_pot_odds(0, 150)

        # Assert
        self.assertEqual(pot_odds, 0)

    def test_get_pot_odds_half_of_pot(self):
        # Act
        pot_odds = self.features.get_pot_odds(75, 150)

        # Assert
        self.assertEqual(pot_odds, 0.5)

    def test_get_pot_odds_same_as_pot(self):
        # Act
        pot_odds = self.features.get_pot_odds(150, 150)

        # Assert
        self.assertEqual(pot_odds, 1)

    def test_get_pot_odds_one_hot_free(self):
        # Act
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(0, 100)

        # Assert
        self.assertEqual(pot_odds_one_hot.tolist(), [1, 0, 0, 0, 0])

    def test_get_pot_odds_one_hot_cheap(self):
        # Act
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(20, 100)

        # Assert
        self.assertEqual(pot_odds_one_hot.tolist(), [0, 1, 0, 0, 0])

    def test_get_pot_odds_one_hot_moderate(self):
        # Act
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(49, 100)

        # Assert
        self.assertEqual(pot_odds_one_hot.tolist(), [0, 0, 1, 0, 0])

    def test_get_pot_odds_one_hot_expensive(self):
        # Act
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(70, 100)

        # Assert
        self.assertEqual(pot_odds_one_hot.tolist(), [0, 0, 0, 1, 0])

    def test_get_pot_odds_one_hot_very_expensive(self):
        # Act
        pot_odds_one_hot = self.features.get_pot_odds_one_hot(76, 100)

        # Assert
        self.assertEqual(pot_odds_one_hot.tolist(), [0, 0, 0, 0, 1])

    def test_get_bet_size(self):
        # Act
        bet_size = self.features.get_bet_size(50, 200)

        # Assert
        self.assertEqual(bet_size, 0.25)

    def test_get_bet_size_one_hot_no_bet(self):
        # Act
        bet_size = self.features.get_bet_size_one_hot(0, 200)

        # Assert
        self.assertEqual(bet_size.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_bet_size_one_hot_all_in(self):
        # Act
        bet_size = self.features.get_bet_size_one_hot(200, 200)

        # Assert
        self.assertEqual(bet_size.tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    """
    Tests to run newly implemented functions.
    These tests dDo not evaluate the actual functionality of the called methods.
    """
    def test_get_hand_probabilities_histogram_example(self):
        # Act
        a = self.features.get_hand_probabilities_histogram(['As', 'Ac'], ['6d', '9d', '2h'])

        str = ''

    def test_get_hand_probabilities_four_of_a_kind(self):
        # Act
        a = self.features.get_hand_probabilities_histogram(['As', 'Ac'], ['Ad', 'Ah', 'Kh'])

        str = ''

    def test_get_hand_probabilities_histogram_flush_draw(self):
        # Act
        a = self.features.get_hand_probabilities_histogram(['Ac', 'Tc'], ['6d', '9c', '2c'])

        str = ''

    def test_get_hand_probabilities_histogram_straight_draw(self):
        # Act
        a = self.features.get_hand_probabilities_histogram(['Js', 'Tc'], ['9c', '8d', '2h'])

        str = ''

    def test_get_hand_probabilities_histogram_one_hot(self):
        # Act
        a = self.features.get_hand_probabilities_histogram_one_hot(['As', 'Ac'], ['6d', '9d', '2h'])

        str = ''
