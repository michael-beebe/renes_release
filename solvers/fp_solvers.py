import copy

import numpy as np


def fictitious_play_strategy(meta_game, max_iterations=int(5e4)):
    n_players = meta_game.shape[0]
    action_dims = meta_game.shape[1:]
    strategies = [np.array([1.0 / i for _ in range(i)]) for i in action_dims]
    # strategies = np.array([([1.0 / i for _ in range(i)]) for i in action_dims])
    # print(type(strategies[0]))
    for cur_iter in range(max_iterations):
        best_responses = []
        for i in range(n_players):
            meta_game_table = copy.deepcopy(meta_game[i])
            for j in range(n_players):
                if i == j:
                    continue
                else:
                    meta_game_table = np.expand_dims(
                        np.average(
                            meta_game_table,
                            axis=j,
                            weights=(strategies[j] / (cur_iter + 1)),
                        ),
                        axis=j,
                    )
            best_responses.append(np.argmax(meta_game_table, axis=i))
        for i in range(n_players):
            strategies[i][best_responses[i]] += 1
    for strategy in strategies:
        strategy /= max_iterations + 1
    return strategies


# meta_games = np.random.rand(2, 3, 3)
#
# meta_games = np.array(
#     [[[0, -1, 1], [1, 0, -1], [-1, 1, 0]], [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]]
# )
# print(meta_games.shape)
# fp_strategies = fictitious_play_strategy(meta_games)
# print(fp_strategies)
# print(meta_games)
#
# strategies = np.random.rand(3)
#
# print(strategies)
#
# print(np.average(meta_games, axis=1, weights=strategies))
#
# update_player = 2
# meta_game = meta_games[2]
# for i in range(3):
#     if i == update_player:
#         continue
#     else:
#         meta_game = np.expand_dims(np.average(meta_game, axis=i, weights=strategies), axis=i)
#     print(meta_game)
# print(meta_game)
