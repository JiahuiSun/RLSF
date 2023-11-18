import os
import argparse
import torch
import time
from tqdm import tqdm
import wandb
import numpy as np

from dataset import EnvDataset
from utils import set_seed
from model import ResNet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='RewardModel')
    parser.add_argument("--device", type=int, default=0, help="cpu if <0, or gpu id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--obs_dim", type=int, default=3)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--data_dir", type=str, default="/home/agent/Code/ackermann_car_nav/data/trajectories/policy_my")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_freq", type=int, default=10)
    args = parser.parse_args()
    return args


def train(args):
    model = ResNet18(args.obs_dim, args.act_dim).to(args.device)

    train_dataloader = torch.utils.data.DataLoader(
        EnvDataset(args.data_dir), batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        EnvDataset(args.data_dir), batch_size=args.batch_size
    )

    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss_list = []
        for traj1_s, traj1_a, traj1_ap, traj2_s, traj2_a, traj2_ap in train_dataloader:
            traj1_s = traj1_s.to(args.device)
            traj1_a = traj1_a.to(args.device)
            traj1_ap = traj1_ap.to(args.device)
            rewards = model(traj1_s, traj1_a)
            traj1_return = torch.sum(rewards, dim=1)

            traj2_s = traj2_s.to(args.device)
            traj2_a = traj2_a.to(args.device)
            traj2_ap = traj2_ap.to(args.device)
            rewards = model(traj2_s, traj2_a)
            traj2_return = torch.sum(rewards, dim=1)

            logits = traj1_return - traj2_return
            targets = ((traj1_ap - traj2_ap) >= 0).float()
            loss = bce_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        model.eval()
        val_loss_list = []
        for traj1_s, traj1_a, traj1_ap, traj2_s, traj2_a, traj2_ap in val_dataloader:
            with torch.no_grad():
                traj1_s = traj1_s.to(args.device)
                traj1_a = traj1_a.to(args.device)
                traj1_ap = traj1_ap.to(args.device)
                rewards = model(traj1_s, traj1_a)
                traj1_return = torch.sum(rewards, dim=1)

                traj2_s = traj2_s.to(args.device)
                traj2_a = traj2_a.to(args.device)
                traj2_ap = traj1_ap.to(args.device)
                rewards = model(traj2_s, traj2_a)
                traj2_return = torch.sum(rewards, dim=1)

                logits = traj1_return - traj2_return
                targets = ((traj1_ap - traj2_ap) >= 0).float()
                loss = bce_loss(logits, targets)
            val_loss_list.append(loss.item())

        wandb.log({
            'train_loss': np.mean(train_loss_list),
            'val_loss': np.mean(val_loss_list),
            'lr': optimizer.param_groups[0]['lr']
        })
        if (epoch+1) % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, f'model-{epoch}.pth'))


if __name__ == '__main__':
    args = get_args()

    set_seed(args.seed)
    args.device = f'cuda:{args.device}' if args.device >=0 and torch.cuda.is_available() else 'cpu'
    logid = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_dir = os.path.join(args.output_dir, logid)
    args.model_dir = os.path.join(log_dir, 'model')
    args.res_dir = os.path.join(log_dir, 'result')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.res_dir, exist_ok=True)
    wandb.init(
        project=f"{args.project_name}",
        name=f"{logid}",
        config=vars(args)
    )

    train(args)
