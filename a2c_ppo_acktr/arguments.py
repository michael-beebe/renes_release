import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
    )
    parser.add_argument(
        "--gail",
        action="store_true",
        default=False,
        help="do imitation learning with gail",
    )
    parser.add_argument(
        "--gail-experts-dir",
        default="./gail_experts",
        help="directory that contains expert demonstrations for gail",
    )
    parser.add_argument(
        "--gail-batch-size",
        type=int,
        default=128,
        help="gail batch size (default: 128)",
    )
    parser.add_argument(
        "--gail-epoch", type=int, default=5, help="gail epochs (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate (default: 7e-4), effective 1e-3",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    # parser.add_argument(
    #     "--alpha",
    #     type=float,
    #     default=0.99,
    #     help="RMSprop optimizer apha (default: 0.99)",
    # )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--use-gae",
        default=True,
        # action="store_true",
        # default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--cuda-deterministic",
        # action="store_true",
        default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=20,
        help="how many training CPU processes to use (default: 16)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="number of forward steps in A2C (default: 5), effective 100",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=16, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=64,
        help="number of batches for ppo (default: 32), effective 64",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=int(6e5),
        help="number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--env-name",
        default="BipedalWalker-v3",
        help="environment to train on (default: PongNoFrameskip-v4)",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/gym/",
        help="directory to save agent logs (default: /tmp/gym)",
    )
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument("--no-cuda", default=False, help="disables CUDA training")
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--recurrent-policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--use-linear-lr-decay",
        # action="store_false",
        default=True,
        help="use a linear schedule on the learning rate",
    )

    # args for game configurations
    # game args
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
        default=30,
        help="The max steps for modifying the game",
    )

    # parser.add_argument(
    #     "--max_episodes",
    #     type=int,
    #     default=int(1e6),
    #     help="The max train episodes of the RL algorithms",
    # )
    #
    # parser.add_argument(
    #     "--eval_episodes",
    #     type=int,
    #     default=int(1e2),
    #     help="The max eval episodes of the RL algorithms",
    # )

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
        "--meta-solver",
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

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ["a2c", "ppo", "acktr"]
    if args.recurrent_policy:
        assert args.algo in [
            "a2c",
            "ppo",
        ], "Recurrent policy is not implemented for ACKTR"

    return args
