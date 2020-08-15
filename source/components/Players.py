class Players:

    def __init__(self):
        pass

    def get_active_players(self, seats, include_own, own_uuid):
        """
        :param seats: All seats taking place in the poker game
        :param include_own: Controls whether the own player is added to the resulting list
        :param own_uuid: Uuid of the own player
        :return: List containing the uuids of the active players
        """
        active_players = []

        for seat in seats:
            if seat['uuid'] == own_uuid and not include_own:
                continue
            if seat['state'] == 'folded':
                continue

            active_players.append(seat['uuid'])

        return active_players

    def get_index_of_player(self, seats, uuid, only_active_players) -> int:
        """
        :param seats: All seats taking place in the poker game
        :param uuid: Uuid of the player for which the index is calculated
        :param only_active_players: Controls if folded players are considered to calculate the index
        :return: Index of the given player
        """
        index_of_own_player = 0
        for seat in seats:
            if seat['uuid'] == uuid:
                return index_of_own_player

            if only_active_players and seat['state'] == 'folded':
                continue

            index_of_own_player += 1

        return index_of_own_player
