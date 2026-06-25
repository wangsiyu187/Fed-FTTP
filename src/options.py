import os
import argparse
import torch

from dataset import shapes_in, shapes_out


def args_parser():
    parser = argparse.ArgumentParser()

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Dataset and Partition Config
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--dataset', type=str, default='odir_multi',
                        # choices=['cifar10', 'cifar100', 'digit', 'pacs_aug'],
                        help='dataset name')

    parser.add_argument('--num_clients', type=int, default=100,
                        help='number of clients')

    parser.add_argument('--partition', type=str, default='step_2_inf',
                        help='how to partition dataset to clients, in format ${method}_${parameters}')

    parser.add_argument('--data_holdout', type=float, default=0.2,
                        help='hold-out rate of data')

    parser.add_argument('--client_holdout', type=float, default=0.2,
                        help='hold-out rate of clients')

    parser.add_argument('--partition_seed', type=int, default=0,
                        help='pre-defined data partition for each client')

    parser.add_argument('--corruption', type=str, default="none",
                        help='none | iid | ood | domain')

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Model Training
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--model', type=str, default='resnet18_multi',
                        choices=['resnet18', 'resnet18_multi', 'resnet50', 'resnet50_multi', 'resnet18_mlgcn', 'cnn'],
                        help='federated learning model')

    parser.add_argument('--loss', type=str, default='ral',
                        choices=['ce', 'bce', 'ral'],
                        help='loss function')

    parser.add_argument('--metric', type=str, default='final',
                        choices=['acc', 'bacc', 'f1', 'kappa', 'auc', 'final'],
                        help='metric function')

    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='the federated learning algorithm')
    
    parser.add_argument('--test_type',type=str,default='off_site',choices=['on_site', 'off_site'],
                        help='Choose which test domain to evaluate on: on_site or off_site.' )

    # Global model

    parser.add_argument('--gm_opt', type=str, default='sgd',
                        help='global model optimizer')

    parser.add_argument('--gm_lr', type=float, default=1.0,
                        help='learning rate of global model optimizer')

    parser.add_argument('--gm_rounds', type=int, default=100,
                        help='number of global communication rounds')

    parser.add_argument('--save_every', type=int, default=0,
                        help='save model checkpoint every N communication rounds (0=disabled, only at end)')

    parser.add_argument('--part_rate', type=float, default=1.0,
                        help='client participation rate in each communication rounds')

    # Local model

    parser.add_argument('--lm_opt', type=str, default='sgd',
                        help='local model optimizer')

    parser.add_argument('--lm_lr', type=float, default=0.1,
                        help='learning rate of the local model optimizer')

    parser.add_argument('--lm_epochs', type=int, default=1,
                        help='number of local training epochs, each epoch iterates the local dataset once')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    
    # Label-shift EM branch switch and hyperparameters

    parser.add_argument('--labelshift', type=str, default='em', choices=['none', 'em'],
                        help='Use calibration+EM label shift at test time.')
    parser.add_argument('--calibration', type=str, default='bvs',
                        choices=['none', 'bvs', 'bcts', 'vs', 'ts'],
                        help='Probability calibration type. bvs=per-class binary vector scaling for BCE; bcts/vs/ts for softmax.')
    parser.add_argument('--em-max-iter', dest='em_max_iter', type=int, default=50)
    parser.add_argument('--em-tol', dest='em_tol', type=float, default=1e-6)
    parser.add_argument('--em-min-prob', dest='em_min_prob', type=float, default=1e-6)

    #RAL

    parser.add_argument('--loss-gamma-pos', dest='loss_gamma_pos', type=float, default=0.0,
                        help='RAL positive focusing gamma (γ+), e.g., 0.0')
    parser.add_argument('--loss-gamma-neg', dest='loss_gamma_neg', type=float, default=3.0,
                        help='RAL negative focusing gamma (γ−), e.g., 3.0')
    parser.add_argument('--loss-tau', dest='loss_tau', type=float, default=0.05,
                        help='RAL negative truncation τ, e.g., 0.05')
    parser.add_argument('--loss-lam', dest='loss_lam', type=float, default=1.5,
                        help='RAL Hill parameter λ, e.g., 1.5')

    parser.add_argument('--loss-M', dest='loss_M', type=int, default=2,
                        help='RAL positive polynomial degree M (>=1), e.g., 2')
    parser.add_argument('--loss-N', dest='loss_N', type=int, default=2,
                        help='RAL negative polynomial degree N (>=1), e.g., 2')

    # Polynomial coefficients: auto-filled to all 1s if not provided (lengths match M, N)
    parser.add_argument('--loss-alpha', dest='loss_alpha', type=float, nargs='*', default=None,
                        help='Coefficients α_1..α_M for positive term (space-separated), e.g., 1 1')
    parser.add_argument('--loss-beta', dest='loss_beta', type=float, nargs='*', default=None,
                        help='Coefficients β_1..β_N for negative term (space-separated), e.g., 1 1')

    # different methods
    parser.add_argument('--prox_mu', type=float, default=0.1,
                    help='mu for FedProx proximal term (0 -> FedAvg)')

    # ======== ML-GCN config ========
    parser.add_argument('--mlgcn_t', type=float, default=0.4,
                        help='threshold t used in ML-GCN adjacency normalization.')
    parser.add_argument('--mlgcn_adj_file', type=str, default='./data/oia_odir/mlgcn_adj_odir_multi.npy',
                        help='path to ML-GCN adjacency matrix (.npy).')
    parser.add_argument('--mlgcn_in_channel', type=int, default=300,
                        help='dimension of label word embeddings (default 300).')


    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Model Adaptation
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    parser.add_argument('--prior_strength', type=float, default=16,
                        help='prior stength for batch norm layers')

    parser.add_argument('--memo_aug_size', type=int, default=16,
                        help='augmentation size for memo')

    parser.add_argument('--t3p_filter_k', type=int, default=-1,
                        help='filter out')

    parser.add_argument('--layers_to_adapt', type=str, default='all',
                        help='which layers to be adapted')

    parser.add_argument('--batchadapt_bn_momentum', type=float, default=0.1,
                        help='batch adapt bn momemtum')

    parser.add_argument('--bn_stat_share_lr', type=str, default='all',
                        # choices=['all', 'block', 'layer', 'none'],
                        help='whether to print a lot')

    parser.add_argument('--grad_norm', type=str, default='sqrt_numel',
                        # choices=['none', 'sqrt_numel', 'numel'],
                        help='how to normalize gradient')

    parser.add_argument('--test', type=str, default='batch',
                        help='batch | online_avg')

    parser.add_argument('--load_adapt_path', type=str, default='none',
                        help='path to load adaptation rates')

    parser.add_argument('--load_adapt_idx', type=int, default=0,
                        help='rank in the pickle file')

    parser.add_argument('--load_adapt_round', type=int, default=-1,
                        help='which round to load')

    parser.add_argument('--eval-thresholds', type=str, default='auto', choices=['auto', '0.5'],
                        help='auto=per-class tau; 0.5=fixed for ablation')
    parser.add_argument('--surgical_metric', type=str, default='valid',
                        help='mode of surgical')

    parser.add_argument('--conf_mode', type=str, default='hard',
                        help='mode of constructing confusion matrix')

    parser.add_argument('--shot_beta', type=float, default=0,
                        help='beta used for shot')

    parser.add_argument('--em_epochs', type=int, default=2,
                        help='beta used for shot')

    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========
    # Other Config
    # ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ========

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    # training
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train ')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers of dataloader')

    # directories
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='where the data is stored')

    parser.add_argument('--data_img', type=str, default='./data/OIA-ODIR_dataset_multi/RGB_preprocessed',
                        help='where the data split.pkl is stored')

    parser.add_argument('--partition_dir', type=str, default='~/data/atp/partition',
                        help='where the data partition is stored')
    parser.add_argument('--partition_path', type=str, default='',
                        help='override partition path (bypasses auto-construction)')

    parser.add_argument('--history_path', type=str, default='none')

    parser.add_argument('--load_model_path', type=str, default='none')

    parser.add_argument('--save_model_path', type=str, default='none')

    # for debug
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to print a lot')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether to visualize ')

    parser.add_argument('--save_gradcam', action='store_true', default=False,
                        help='save Grad-CAM visualization images')
# Multi-label
    parser.add_argument('--label_txt', type=str, default="./data/oia_odir/train_labels.txt",
        help='ODIR multi-label txt (when dataset=odir_multi)')

    args = parser.parse_args()

    # RAL: normalize alpha/beta only when needed
    if args.loss == 'ral':
        # pad/truncate alpha to length M
        if args.loss_alpha is None or len(args.loss_alpha) == 0:
            args.loss_alpha = [1.0] * args.loss_M
        else:
            args.loss_alpha = list(map(float, args.loss_alpha))
            if len(args.loss_alpha) < args.loss_M:
                args.loss_alpha += [1.0] * (args.loss_M - len(args.loss_alpha))
            elif len(args.loss_alpha) > args.loss_M:
                args.loss_alpha = args.loss_alpha[:args.loss_M]

        # pad/truncate beta to length N
        if args.loss_beta is None or len(args.loss_beta) == 0:
            args.loss_beta = [1.0] * args.loss_N
        else:
            args.loss_beta = list(map(float, args.loss_beta))
            if len(args.loss_beta) < args.loss_N:
                args.loss_beta += [1.0] * (args.loss_N - len(args.loss_beta))
            elif len(args.loss_beta) > args.loss_N:
                args.loss_beta = args.loss_beta[:args.loss_N]

        print(
            "[LOSS] Using RAL "
            f"(gamma_pos={args.loss_gamma_pos}, gamma_neg={args.loss_gamma_neg}, "
            f"tau={args.loss_tau}, lam={args.loss_lam}, "
            f"M={args.loss_M}, N={args.loss_N}, "
            f"alpha={args.loss_alpha}, beta={args.loss_beta})"
        )
    else:
        print(f"[LOSS] Using {args.loss.upper()}")

    # number of clients
    args.num_train_clients = round((1 - args.client_holdout) * args.num_clients)
    args.num_test_clients = args.num_clients - args.num_train_clients
    # in and out-dimension of model
    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)  # binary classification has one output

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)

    # the path of partition config (skip if explicitly overridden)
    if not args.partition_path:
        if args.partition_seed is None:
            args.partition_seed = args.seed
        partition_filename = 'client_%d_partition_%s_seed_%d.pkl' % (
            args.num_clients, args.partition, args.partition_seed)
        args.partition_path = os.path.join(args.partition_dir, args.dataset, partition_filename)

    # the path for corrupted dataset
    corruption_filename = 'client_%d_partition_%s_corruption_%s_seed_%d.pkl' % (
        args.num_clients, args.partition, args.corruption, args.partition_seed)
    args.corruption_path = os.path.join(args.data_dir, 'atp', args.dataset, corruption_filename)

    # the path of domain dataset
    if args.dataset == 'pacs_aug':
        args.domain_path = os.path.join(args.data_dir, 'atp',
                                        args.dataset + '_seed_' + str(args.partition_seed) + '.pkl')

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args
