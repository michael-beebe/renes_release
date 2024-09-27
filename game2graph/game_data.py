import numpy as np
from tensorly.decomposition import parafac

from game2graph.response_graphs import (
    alpha_rank_response_graph,
    alpha_rank_response_graph_inf_alpha,
)
from utils import normalize_tables


def gen_game_datasets(args):
    train_number = args.train_number
    test_number = args.test_number

    train_dataset = []
    test_dataset = []

    for i in range(train_number):
        payoff_tables, weights, factors = gen_game(args)
        train_dataset.append((payoff_tables, weights, factors))

    for i in range(test_number):
        payoff_tables, weights, factors = gen_game(args)
        test_dataset.append((payoff_tables, weights, factors))

    return train_dataset, test_dataset


def gen_game(args):
    min_players = args.min_players
    max_players = args.max_players
    max_actions = args.max_actions
    min_actions = args.min_actions
    # m = args.m
    # alpha = args.alpha
    # use_inf_alpha = args.use_inf_alpha
    num_players = np.random.randint(low=min_players, high=max_players + 1)
    # print(num_players)
    action_dims = [
        np.random.randint(low=min_actions, high=max_actions + 1)
        for _ in range(num_players)
    ]

    payoff_tables, weights, factors = None, None, None
    while True:
        try:
            payoff_tables = np.random.random([num_players] + action_dims)
            payoff_tables = normalize_tables(
                tables=payoff_tables, max_val=args.max_val, min_val=args.min_val
            )
            # print("tensor decomp start")
            # payoff_tables = torch.from_numpy(payoff_tables)

            (weights, factors) = parafac(
                payoff_tables,
                init="random",
                rank=args.action_size,
                n_iter_max=int(1e3),
                tol=1.0e-5,
                linesearch=True,
            )
        except:
            continue
        if np.isnan(weights).any():
            continue

        flag = False
        for factor in factors:
            if np.isnan(factor).any():
                flag = True
                break
        if flag:
            continue
        break

    return payoff_tables, weights, factors


def gen_response_graph(payoff_tables, args):
    m = args.m
    alpha = args.alpha
    use_inf_alpha = args.use_inf_alpha
    if use_inf_alpha:
        result_dict = alpha_rank_response_graph_inf_alpha(payoff_tables)
    else:
        result_dict = alpha_rank_response_graph(
            payoff_tables, m=m, alpha=alpha, use_inf_alpha=False
        )

    return result_dict
