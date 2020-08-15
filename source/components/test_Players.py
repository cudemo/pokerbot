import unittest

from components.Players import Players


class TestPlayers(unittest.TestCase):

    def setUp(self):
        self.players = Players()

    def test_get_active_players_include_own(self):
        # Arrange
        seats = [
            {'uuid': 'aaaaaaaaaaaaaaaaaaaaaa', 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
            {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
        ]

        # Act
        active_players = self.players.get_active_players(seats, True, 'aaaaaaaaaaaaaaaaaaaaaa')

        # Assert
        self.assertEqual(active_players, ['aaaaaaaaaaaaaaaaaaaaaa', 'bbbbbbbbbbbbbbbbbbbbbb', 'cccccccccccccccccccccc'])

    def test_get_active_players_exclude_own(self):
        # Arrange
        seats = [
            {'uuid': 'aaaaaaaaaaaaaaaaaaaaaa', 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'participating'},
            {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
        ]

        # Act
        active_players = self.players.get_active_players(seats, False, 'aaaaaaaaaaaaaaaaaaaaaa')

        # Assert
        self.assertEqual(active_players, ['bbbbbbbbbbbbbbbbbbbbbb', 'cccccccccccccccccccccc'])

    def test_get_index_of_player_all_players(self):
        # Arrange
        seats = [
            {'uuid': 'aaaaaaaaaaaaaaaaaaaaaa', 'state': 'participating'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'folded'},
            {'uuid': 'cccccccccccccccccccccc', 'state': 'folded'}
        ]

        # Act
        index = self.players.get_index_of_player(seats, 'cccccccccccccccccccccc', False)

        # Assert
        self.assertEqual(index, 2)

    def test_get_index_of_player_only_actives(self):
        # Arrange
        seats = [
            {'uuid': 'aaaaaaaaaaaaaaaaaaaaaa', 'state': 'folded'},
            {'uuid': 'bbbbbbbbbbbbbbbbbbbbbb', 'state': 'folded'},
            {'uuid': 'cccccccccccccccccccccc', 'state': 'participating'}
        ]

        # Act
        index = self.players.get_index_of_player(seats, 'cccccccccccccccccccccc', True)

        # Assert
        self.assertEqual(index, 0)
