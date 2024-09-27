import copy

import numpy as np


def nash_conv(meta_game, strategies):
    ZERO = 1e-9
    n_players = meta_game.shape[0]
    action_dims = meta_game.shape[1:]
    # strategies = np.array([[1.0 / i for _ in range(i)] for i in action_dims])
    best_responses = []
    game_vectors = []
    exploitabilities = []
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
                        weights=(strategies[j]),
                    ),
                    axis=j,
                )
        best_responses.append(np.argmax(meta_game_table, axis=i))
        game_vectors.append(meta_game_table)
        best_utility = np.max(meta_game_table, axis=i)
        mean_utility = np.average(
            meta_game_table,
            axis=i,
            weights=(strategies[i]),
        )
        # print(best_utility.shape)
        # print(mean_utility)
        for _ in range(n_players):
            best_utility = np.squeeze(best_utility, axis=0)
            mean_utility = np.squeeze(mean_utility, axis=0)
        exploitability = best_utility - mean_utility
        exploitabilities.append(exploitability if exploitability > ZERO else 0)
    return exploitabilities
