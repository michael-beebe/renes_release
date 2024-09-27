import numpy as np
import open_spiel.python.egt.utils as utils
import torch
from torch_geometric.data import Data

from solvers import alpha_rank


#######################################################################
# generate the response graph
#######################################################################


def unnormalized_response_graph(meta_games):
    meta_games = [np.asarray(x) for x in meta_games]
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(meta_games)


def alpha_rank_response_graph(meta_games, m, alpha, use_inf_alpha):
    payoffs_are_hpt_format = utils.check_payoffs_are_hpt(meta_games)
    c, _ = alpha_rank.get_multipop_transition_matrix(
        meta_games,
        payoffs_are_hpt_format,
        m=m,
        alpha=alpha,
        use_inf_alpha=use_inf_alpha,
    )
    # strat_labels = utils.get_strat_profile_labels(meta_games, payoffs_are_hpt_format)
    num_strats_per_population = utils.get_num_strats_per_population(
        meta_games, payoffs_are_hpt_format
    )
    node_to_strat_labels = []
    for i in range(len(c)):
        node_to_strat_labels.append(
            utils.get_strat_profile_from_id(num_strats_per_population, i)
        )
    return {"markov_transition_matrix": c, "node_to_strat_labels": node_to_strat_labels}


def alpha_rank_response_graph_inf_alpha(meta_games):
    return alpha_rank_response_graph(meta_games, m=50, alpha=10, use_inf_alpha=True)


###################################################################
# pre process the alpha rank response graph to gnn
# node feature, edge index, and edge attribute
###################################################################


def preprocess_response_graph_to_gnn(response_graph_data, max_players=4, max_actions=5):
    ZERO = 1e-14

    payoff_tables = response_graph_data["payoff_tables"]
    adj_matrix = response_graph_data["markov_transition_matrix"]
    node_strat = response_graph_data["node_to_strat_labels"]

    node_number = len(node_strat)
    num_players = len(payoff_tables)
    com_players = max_players - num_players

    # print(node_number)
    strat_val = []
    for strat in node_strat:
        val = []
        # print(type(strat))
        # print([0] + strat)
        for i in range(num_players):
            val.append(payoff_tables[tuple([i] + list(strat))])
        strat_val.append(val)
    node_strat = np.array(node_strat)
    strat_val = np.array(strat_val)
    # print(strat_val.shape)
    complementary_feature = np.zeros([node_number, com_players])
    aug_node_feature = np.concatenate(
        [node_strat, complementary_feature, strat_val, complementary_feature], axis=-1
    )

    edge_index = []
    edge_attr = []
    adj_matrix[adj_matrix < ZERO] = 0
    (row, col) = adj_matrix.shape
    for i in range(row):
        for j in range(col):
            if adj_matrix[i][j]:
                edge_index.append([i, j])
                edge_attr.append([adj_matrix[i][j]])
    edge_index = np.array(edge_index).transpose()

    aug_node_feature = torch.from_numpy(aug_node_feature).type(torch.float32)
    edge_index = torch.from_numpy(edge_index).type(torch.long)
    edge_attr = torch.from_numpy(np.array(edge_attr)).type(torch.float32)
    data = Data(x=aug_node_feature, edge_index=edge_index, edge_attr=edge_attr)

    return data


def preprocess_response_graph_to_gnn_no_max(response_graph_data):
    ZERO = 1e-14

    adj_matrix = response_graph_data["markov_transition_matrix"]

    edge_index = []
    edge_attr = []
    adj_matrix[adj_matrix < ZERO] = 0
    (row, col) = adj_matrix.shape
    for i in range(row):
        for j in range(col):
            if adj_matrix[i][j]:
                edge_index.append([i, j])
                edge_attr.append([adj_matrix[i][j]])
    edge_index = np.array(edge_index).transpose()

    node_feature = np.ones([row, 1])
    node_feature = torch.from_numpy(node_feature).type(torch.float32)
    edge_index = torch.from_numpy(edge_index).type(torch.long)
    edge_attr = torch.from_numpy(np.array(edge_attr)).type(torch.float32)
    data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr)

    return data


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
