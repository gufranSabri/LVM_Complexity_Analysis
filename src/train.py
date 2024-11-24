import os
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scheduler import LinearDecayLR
from utils import *

def validation(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, total=len(loader), desc=f"valid"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct, total = 0, 0
    for inputs, targets in tqdm(loader, total=len(loader), desc="train"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

def train(args):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == 'mps':
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic=True
        torch.backends.mps.benchmark = False
    elif args.device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    args.output_dir = os.path.join(args.output_dir,f"{args.model}_{args.phase}_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    model_path = os.path.join(args.output_dir, f"model.pth")
    log_path = os.path.join(args.output_dir, f"logs.log")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model = get_model(args.model, pretrained=args.phase=="2").to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) \
        if args.phase == "2" \
        else optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.999))
    lr_scheduler=LinearDecayLR(optimizer, int(args.epochs), int(int(args.epochs)*0.25))

    train_loader, test_loader = get_CIFAR_loaders(args.batch_size, args.batch_size) \
        if args.phase == "2" \
        else get_ImageNet1K_loaders(args.batch_size, args.batch_size)
    
    train_logger = Logger(log_path)
    train_logger(f"Model: {args.model}, Phase: {args.phase}")
    train_logger(f"Batch Size: {args.batch_size}, Learning Rate: {args.lr}, Epochs: {args.epochs}")
    train_logger(f"Output Directory: {args.output_dir}")
    train_logger(f"Device: {args.device}")
    train_logger("=========================================")

    for epoch in range(int(args.epochs)):
        train_logger(f"\nEpoch [{epoch+1}/{args.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device=args.device)
        val_loss, val_acc = validation(model, test_loader, criterion, device=args.device)

        train_logger(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        train_logger(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
        train_logger(f"Learning Rate: {lr_scheduler.get_lr()}")
        lr_scheduler.step()

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformers on CIFAR-100")
    parser.add_argument("--model", type=str, choices=["vit", "deit", "swin", "resnet"], required=True, help="Model type")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default="50", help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints")
    parser.add_argument("--phase", type=str, default="2", help="1: Pretrain, 2: Finetune")
    parser.add_argument("--device", type=str, default="cuda")    
    args = parser.parse_args()

    train(args)