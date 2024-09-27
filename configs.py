import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument(
        "--min_players", type=int, default=2, help="The max number of players"
    )
    parser.add_argument(
        "--max_players", type=int, default=4, help="The max number of players"
    )
    parser.add_argument(
        "--max_actions", type=int, default=5, help="The max actions of players"
    )
    parser.add_argument(
        "--min_actions", type=int, default=4, help="The max actions of players"
    )

    parser.add_argument(
        "--min_val", type=float, default=-5, help="The min val of payoff"
    )

    parser.add_argument(
        "--max_val", type=float, default=5, help="The max val of payoff"
    )

    parser.add_argument(
        "--train_number",
        type=int,
        default=5000,
        help="The number of games used in training",
    )
    parser.add_argument(
        "--test_number",
        type=int,
        default=200,
        help="The number of games used in testing",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="The max steps for modifying the game",
    )

    parser.add_argument(
        "--max_episodes",
        type=int,
        default=int(1e6),
        help="The max train episodes of the RL algorithms",
    )

    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=int(1e2),
        help="The max eval episodes of the RL algorithms",
    )

    parser.add_argument(
        "--action_size",
        type=int,
        default=10,
        help="The max actions when use decomposition",
    )

    parser.add_argument(
        "--decomp_or_not",
        type=bool,
        default=True,
        help="Whether to use the decomposition of the payoff tensor",
    )

    parser.add_argument(
        "--weight_step",
        type=float,
        default=5.0,
        help="the step size of the update of the matrix weight",
    )

    parser.add_argument(
        "--use_inf_alpha",
        type=bool,
        default=False,
        help="Whether to use the infinite alpha in the alpha rank response graph",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=5,
        help="The parameter m in the alpha rank response graph",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="The parameter alpha in the alpha rank response graph",
    )

    parser.add_argument(
        "--meta_solver",
        type=str,
        default="alpha_rank",
        help="the meta solver",
    )

    # node_output_size
    parser.add_argument(
        "--node_output_size",
        type=int,
        default=20,
        help="The output feature of GNN encoder",
    )

    parser.add_argument(
        "--gnn_layers",
        type=int,
        default=2,
        help="The number of layers in GNN encoder",
    )

    return parser
