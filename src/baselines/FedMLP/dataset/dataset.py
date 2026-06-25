import os
import pickle

import numpy as np
from torchvision.transforms import transforms
from dataset.atp_adapter import load_atp_data, ATPClientDataset, ATPWrapperDataset

from dataset.all_dataset import ChestXray14, ICH, ODIR
from utils.FixMatch import RandAugmentMC
from utils.sampling import iid_sampling, non_iid_dirichlet_sampling


def get_dataset(args):
    if args.dataset == "ChestXray14":
        root = "/home/szb/multilabel/onehot-label-PA.csv"
        args.n_classes = 8
        args.n_clients = 8
        args.num_users = args.n_clients
        args.input_channel = 3

        # normalize = transforms.Normalize([0.498, 0.498, 0.498],
        #                                  [0.228, 0.228, 0.228])
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if args.exp == 'FedAVG' or args.exp == 'RoFL' or args.exp == 'FedNoRo' or args.exp == 'CBAFed':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ChestXray14(root, "train", train_transform)
            test_dataset = ChestXray14(root, "test", test_transform)

        elif args.exp == 'RSCFed' or args.exp == 'FedPN' or args.exp == 'FedLSR' or args.exp == 'FedIRM':
            train_transform1 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform2 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ChestXray14(root, "train", (train_transform1, train_transform2))
            test_dataset = ChestXray14(root, "test", test_transform)

        elif args.exp == 'FedAVG+FixMatch':
            train_weak_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_strong_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ChestXray14(root, "train", (train_weak_transform, train_strong_transform))
            test_dataset = ChestXray14(root, "test", test_transform)

    elif args.dataset == "ICH":
        root = "/home/szb/ICH_stage2/ICH_stage2/data_png185k_512.csv"
        args.n_classes = 5
        args.n_clients = 5
        args.num_users = args.n_clients
        args.input_channel = 3

        # normalize = transforms.Normalize([0.498, 0.498, 0.498],
        #                                  [0.228, 0.228, 0.228])
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if args.exp == 'FedAVG' or args.exp == 'RoFL' or args.exp == 'FedNoRo' or args.exp == 'CBAFed':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ICH(root, "train", train_transform)
            test_dataset = ICH(root, "test", test_transform)

        elif args.exp == 'RSCFed' or args.exp == 'FedPN' or args.exp == 'FedLSR' or args.exp == 'FedIRM':
            train_transform1 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform2 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ICH(root, "train", (train_transform1, train_transform2))
            test_dataset = ICH(root, "test", test_transform)

        elif args.exp == 'FedAVG+FixMatch':
            train_weak_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_strong_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ICH(root, "train", (train_weak_transform, train_strong_transform))
            test_dataset = ICH(root, "test", test_transform)

    elif args.dataset == "ODIR":
        args.n_classes = 8
        args.n_clients = 5
        args.num_users = args.n_clients
        args.input_channel = 3

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if args.exp == 'FedAVG' or args.exp == 'RoFL' or args.exp == 'FedNoRo' or args.exp == 'CBAFed':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ODIR(args.data_dir, args.label_txt, "train", train_transform)
            test_dataset_offsite = ODIR(args.data_dir, args.label_txt, "test_offsite", test_transform, args.corruption_std)
            test_dataset_onsite = ODIR(args.data_dir, args.label_txt, "test_onsite", test_transform, args.corruption_std)

        elif args.exp == 'RSCFed' or args.exp == 'FedPN' or args.exp == 'FedLSR' or args.exp == 'FedIRM' or args.exp == 'FedMLP':
            train_transform1 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_transform2 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ODIR(args.data_dir, args.label_txt, "train", (train_transform1, train_transform2))
            test_dataset_offsite = ODIR(args.data_dir, args.label_txt, "test_offsite", test_transform, args.corruption_std)
            test_dataset_onsite = ODIR(args.data_dir, args.label_txt, "test_onsite", test_transform, args.corruption_std)

        elif args.exp == 'FedAVG+FixMatch':
            train_weak_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            train_strong_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
            train_dataset = ODIR(args.data_dir, args.label_txt, "train", (train_weak_transform, train_strong_transform))
            test_dataset_offsite = ODIR(args.data_dir, args.label_txt, "test_offsite", test_transform, args.corruption_std)
            test_dataset_onsite = ODIR(args.data_dir, args.label_txt, "test_onsite", test_transform, args.corruption_std)

    elif args.dataset == "ODIR_ATP":
        # ATP-aligned: step_2_16 partition + gaussian_noise corruption
        args.n_classes = 8
        args.n_clients = 5
        args.num_users = args.n_clients
        args.input_channel = 3
        args.pretrained = True  # ATP uses ImageNet pretrained ResNet18
        args.feature_dim = 512

        # Use on-site pickle if requested
        if args.test_set == "onsite":
            args.atp_corruption_file = args.atp_corruption_file.replace('.pkl', '_onsite.pkl')
            args.atp_partition_file = args.atp_partition_file.replace('.pkl', '_onsite.pkl')

        corruption_path = os.path.join(args.atp_data_dir, args.atp_corruption_file)
        partition_path = os.path.join(args.atp_partition_dir, args.atp_partition_file)
        X, Y, train_clients, test_clients = load_atp_data(corruption_path, partition_path)

        dict_users = {}
        for cid in sorted(train_clients.keys()):
            dict_users[cid] = train_clients[cid]['train'].tolist()

        # All training samples (indices 0-6973). DatasetSplit uses dict_users to filter per-client.
        train_indices = list(range(6974))
        train_dataset = ATPClientDataset(X, Y, train_indices, mode='train')

        test_cid = sorted(test_clients.keys())[0]
        test_indices = test_clients[test_cid]['test'].tolist()
        test_dataset_offsite = ATPWrapperDataset(X, Y, test_indices)
        test_dataset_onsite = None

    else:
        exit("Error: unrecognized dataset")

    n_train = len(train_dataset)
    y_train = np.array(train_dataset.targets)
    assert n_train == len(y_train)
    print(n_train)

    # Load or Generate 'dict_users' (skip for ATP-aligned which uses pre-defined partition)
    if args.dataset != "ODIR_ATP":
        if args.iid == 0:   # non-iid
            if os.path.exists(f"non-iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients)+'_'+str(args.alpha_dirichlet)}.npy"):
                dict_users = np.load(f"non-iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients)+'_'+str(args.alpha_dirichlet)}.npy", allow_pickle=True).item()
            else:
                dict_users = non_iid_dirichlet_sampling(y_train, args.n_classes, 1.0, args.n_clients, seed=args.seed, alpha_dirichlet=args.alpha_dirichlet)
            np.save(f"non-iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients)+'_'+str(args.alpha_dirichlet)}.npy", dict_users, allow_pickle=True)
        else:
            if os.path.exists(f"iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients) + '5000'}.npy"):
                dict_users = np.load(f"iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients) + '5000'}.npy", allow_pickle=True).item()
            else:
                dict_users = iid_sampling(n_train, args.n_clients, args.seed)
                np.save(f"iid-dictusers/{str(args.dataset)+'_'+str(args.seed)+'_'+str(args.n_clients) + '5000'}.npy", dict_users, allow_pickle=True)
    if args.dataset == "ODIR" or args.dataset == "ODIR_ATP":
        return train_dataset, test_dataset_offsite, test_dataset_onsite, dict_users
    else:
        return train_dataset, test_dataset, None, dict_users


