import argparse


def get_arguments():
    parser = argparse.ArgumentParser('Arguments for this program')

    # [Main arguments] for reproducibility and different paths
    parser.add_argument('--experiment_id',
                        type=str,
                        default="0",
                        help='Experiment ID')
    parser.add_argument('--note',
                        type=str,
                        default='',
                        help='Appends to experiment id')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2023,
                        help="Random seed for reproducibility")

    parser.add_argument('--logs_dir',
                        type=str,
                        default='logs/',
                        help="Path to save logs")
    parser.add_argument('--save_dir',
                        type=str,
                        default='results/'
                        help="Path to save results and checkpoints")

    parser.add_argument('--load',
                        action='store_true',
                        default=None,
                        help="Whether to load model")
    parser.add_argument('--batch',
                        action='store_true',
                        default=None,
                        help="To load the last batch model")
    parser.add_argument('--best',
                        action='store_true',
                        default=None,
                        help="To load the best model")

    # [Data] related arguments
    parser.add_argument('--train_data_path',
                        type=str,
                        default='data/train_data.csv',
                        help="Path to training data")
    parser.add_argument('--test_data_path',
                        type=str,
                        default='data/test_data.csv',
                        help='Path to testing data')

    # [Inference]
    parser.add_argument('--n_epochs_start_viz',
                        type=int,
                        default=0,
                        help="When to start vizualization")
    parser.add_argument('--n_epochs_to_viz',
                        type=int,
                        default=1,
                        help="Vizualize every N epochs")

    parser.add_argument('--test_only',
                        action='store_true',
                        help='Whether only to test, no training')
    parser.add_argument('--make_plots',
                        action='store_true',
                        help='Whether to make plots in inference')
    parser.add_argument('--make_summary',
                        action='store_true',
                        help='Whether to commpute summary in inference')
    parser.add_argument(
        '--check_memory_time',
        action='store_true',
        help='Whether to print check train time and gpu memory')

    # [Model] training
    parser.add_argument('--device', type=int, default=0, help='Cuda device')
    parser.add_argument(
        '--n_data',
        type=int,
        default=None,
        help="Number of samples to use (if not None), mainly for debugging.")
    parser.add_argument(
        '--train_perc',
        type=float,
        default=1,
        help="Size of the training, the rest goes to validation")
    parser.add_argument('--shuffle',
                        action='store_true',
                        help="Whether to shuffle training data")
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help="Starting learning rate.")
    parser.add_argument(
        '--reset_lr',
        action='store_true',
        help="To reload LR loaded from the model on default value")

    # 1. Args definining neural networks
    parser.add_argument('--model',
                        type=str,
                        default='rnn',
                        help="Temporal model")
    parser.add_argument('--n_layers',
                        type=int,
                        default=1,
                        help="Number of layers for temporal model")
    parser.add_argument('--n_hidden',
                        type=int,
                        default=100,
                        help="Number of units per layer in temporal model")
    parser.add_argument(
        '--p_dropout',
        type=float,
        default=0.3,
        help="Probability of dropout (anywhere in architecture)")

    return parser
