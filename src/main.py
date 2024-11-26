import os
import time
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
from metrics import *

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
    # set seed ===============================================
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
    # set seed ===============================================

    # create output directory ================================
    args.output_dir = os.path.join(args.output_dir,f"{args.model}_{args.phase}_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    model_path = os.path.join(args.output_dir, f"model.pth")
    log_path = os.path.join(args.output_dir, f"train.log")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # create output directory ================================

    # initialize components ==================================
    model = get_model(args.model, pretrained=args.phase=="2").to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) \
        if args.phase == "2" \
        else optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.999))
    lr_scheduler=LinearDecayLR(optimizer, int(args.epochs), int(int(args.epochs)*0.25))
    # initialize components ==================================

    # initialize dataloaders =================================
    train_loader, test_loader = get_CIFAR_loaders(int(args.batch_size), int(args.batch_size)) \
        if args.phase == "2" \
        else get_ImageNet1K_loaders(int(args.batch_size), int(args.batch_size))
    # initialize dataloaders =================================
    
    # log training details ===================================
    train_metrics_logger = Logger(log_path)
    train_metrics_logger(f"Model: {args.model}, Phase: {args.phase}")
    train_metrics_logger(f"Batch Size: {int(args.batch_size)}, Learning Rate: {args.lr}, Epochs: {args.epochs}")
    train_metrics_logger(f"Output Directory: {args.output_dir}")
    train_metrics_logger(f"Device: {args.device}")
    train_metrics_logger("=========================================")
    # log training details ===================================

    total_train_time = 0
    for epoch in range(int(args.epochs)):
        start_time = time.time()
        train_metrics_logger(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device=args.device)
        total_train_time += time.time() - start_time

        val_loss, val_acc = validation(model, test_loader, criterion, device=args.device)

        # log training metrics ===============================
        train_metrics_logger(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        train_metrics_logger(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
        train_metrics_logger(f"Learning Rate: {lr_scheduler.get_lr()}")
        # log training metrics ===============================

        lr_scheduler.step()

    # measure complexity metrics =============================
    per_instance_inference_latency = measure_per_instance_inference_latency(model, args.device)
    per_batch_gpu_memory_consumption_b4 = measure_per_batch_gpu_memory_consumption(model, 4, args.device)
    per_batch_gpu_memory_consumption_b8 = measure_per_batch_gpu_memory_consumption(model, 8, args.device)
    per_batch_gpu_memory_consumption_b16 = measure_per_batch_gpu_memory_consumption(model, 16, args.device)
    per_batch_gpu_memory_consumption_b32 = measure_per_batch_gpu_memory_consumption(model, 32, args.device)
    per_batch_gpu_memory_consumption_b64 = measure_per_batch_gpu_memory_consumption(model, 64, args.device)
    per_batch_gpu_memory_consumption_b128 = measure_per_batch_gpu_memory_consumption(model, 128, args.device)
    per_batch_gpu_memory_consumption_b256 = measure_per_batch_gpu_memory_consumption(model, 256, args.device)
    val_loss, val_acc = validation(model, test_loader, criterion, device=args.device)
    # measure complexity metrics =============================

    # log complexity metrics =================================
    complexity_stats_logger = Logger(log_path.replace("train","stats"))
    complexity_stats_logger(f"Model: {get_model_name(args.model)}")
    complexity_stats_logger(f"Dataset size: {len(train_loader.dataset)}")
    complexity_stats_logger(f"Batch Size: {args.batch_size}")
    complexity_stats_logger(f"Average time per forward pass: {total_train_time/int(args.epochs)/len(train_loader)/int(args.batch_size)}")
    complexity_stats_logger(f"Training time per epoch: {total_train_time/int(args.epochs)}")
    complexity_stats_logger(f"Per instance inference latency: {per_instance_inference_latency}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 4): {per_batch_gpu_memory_consumption_b4}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 8): {per_batch_gpu_memory_consumption_b8}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 16): {per_batch_gpu_memory_consumption_b16}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 32): {per_batch_gpu_memory_consumption_b32}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 64): {per_batch_gpu_memory_consumption_b64}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 128): {per_batch_gpu_memory_consumption_b128}")
    complexity_stats_logger(f"Per batch GPU memory consumption (batch size 256): {per_batch_gpu_memory_consumption_b256}")
    complexity_stats_logger(f"Validation Loss: {val_loss:.4f}")
    complexity_stats_logger(f"Validation Accuracy: {val_acc:.2f}%")
    # log complexity metrics =================================

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformers on CIFAR-100")
    parser.add_argument("--model", type=str, choices=["vit", "deit", "swin", "resnet"], required=True, help="Model type")
    parser.add_argument("--batch_size", type=int, default="128", help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default="50", help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save checkpoints")
    parser.add_argument("--phase", type=str, default="2", help="1: Pretrain, 2: Finetune")
    parser.add_argument("--device", type=str, default="cuda")    
    args = parser.parse_args()

    train(args)