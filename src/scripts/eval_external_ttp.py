
import os
import torch
from torch.utils.data import DataLoader

from options import args_parser
from model import create_model, create_loss
from ttp.atp import atp_adapt  

from dataset.external_odir_datasets import DDRAsODIR, REFUGEAsODIR
from scripts.eval_external import eval_binary_subset


def run_ttp_adapt(model, loader, device):
    """
    Execute test-time adaptation ATP
    """
    print("=> Running Test-time Personalization (ATP)...")
    model.train()


    atp_adapt(model, loader, device=device)

    return model


def main():
    args = args_parser()

    args.dataset = 'odir_multi'
    args.model = 'resnet18_multi'
    args.loss = 'ral'
    args.shape_in = (3, 256, 256)
    args.shape_out = 8
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load OIA-ODIR pretrained model
    args.load_model_path = (
        "./"
        "weights/odir_multi/label/"
        "pretrain_fedavg_ral_resnet18_multi__pseed_0_seed_0.pkl"
    )

    print(f"=> Loading model from: {args.load_model_path}")
    model = create_model(args)
    criterion = create_loss(args)

    # ===================== DDR ===========================
    ddr_root = "REPLACE_ME/DDR-dataset/DR_grading"

    ddr_test = DDRAsODIR(
        img_root=os.path.join(ddr_root, "test"),
        split_txt=os.path.join(ddr_root, "test.txt"),
        resize=256,
    )
    ddr_loader = DataLoader(ddr_test, batch_size=64, shuffle=False, num_workers=4)
    print(f"[DDR] samples = {len(ddr_test)}")

    # First compute baseline
    base_metrics = eval_binary_subset(
        model, ddr_loader, criterion, args.device, pos_idx=1, neg_idx=0
    )

    # Then execute TTP training, including validation
    model_ttp = run_ttp_adapt(model, ddr_loader, args.device)

    ttp_metrics = eval_binary_subset(
        model_ttp, ddr_loader, criterion, args.device, pos_idx=1, neg_idx=0
    )

    print("=== DDR Results ===")
    print("Before-TTP:", base_metrics)
    print("After-TTP :", ttp_metrics)

    # ===================== REFUGE ===========================
    refuge_root = "REPLACE_ME/REFUGE-Multirater"

    refuge_test = REFUGEAsODIR(
        refuge_root=refuge_root,
        csv_path=os.path.join(refuge_root, "REFUGE1Test.csv"),
        resize=256,
    )
    refuge_loader = DataLoader(refuge_test, batch_size=64, shuffle=False, num_workers=4)
    print(f"[REFUGE] samples = {len(refuge_test)}")

    # baseline
    # This model is a newly created model during TTP
    model = create_model(args)

    base_metrics = eval_binary_subset(
        model, refuge_loader, criterion, args.device, pos_idx=2, neg_idx=0
    )

    # TTP
    model_ttp = run_ttp_adapt(model, refuge_loader, args.device)

    ttp_metrics = eval_binary_subset(
        model_ttp, refuge_loader, criterion, args.device, pos_idx=2, neg_idx=0
    )

    print("=== REFUGE Results ===")
    print("Before-TTP:", base_metrics)
    print("After-TTP :", ttp_metrics)


if __name__ == "__main__":
    main()
