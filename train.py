import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import MidiEmbeddingDataset
from models.MidiResNet import MidiResNet
from models.MidiTransNet import MidiTransNet
from loss import ControlledMusicEmbeddingLoss
import logging
import time
from tqdm import tqdm
import argparse
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_logger(work_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='[%(asctime)s|%(filename)s|%(levelname)s] %(message)s',
        datefmt='%a %b %d %H:%M:%S %Y',
    )

    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger

def train(model, train_loader, optimizer, scheduler, loss_fn, device, logger, pitch_only=True):
    """
    Train the model for one epoch.

    Args:
        model: The embedding model.
        train_loader: DataLoader for the training set.
        optimizer: Optimizer for the model.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function to use.
        device: Device to run the computation on.
        logger: Logger to record the loss.
        pitch_only: Whether to use pitch-only data.

    Returns:
        loss_avg: Average loss over the epoch.
    """
    model.train()
    train_nb = len(train_loader)
    loss_accumlation = loss_avg = 0
    pbar_train = tqdm(train_loader, total=train_nb, desc='Train')
    
    for step, data in enumerate(pbar_train):
        # if step == 10:
        #     break

        # Move data to device
        if pitch_only:
            anchor_pitch, positive_pitch, ref = data
        else:
            anchor_pitch, positive_pitch, anchor_duration, positive_duration, ref = data
            anchor_duration = anchor_duration.to(device)
            positive_duration = positive_duration.to(device)
        
        anchor_pitch = anchor_pitch.to(device)
        positive_pitch = positive_pitch.to(device)
        ref = ref.to(device) 

        # Forward pass
        anchor_embed = model(anchor_pitch) if pitch_only else model(anchor_pitch, anchor_duration)
        positive_embed = model(positive_pitch) if pitch_only else model(positive_pitch, positive_duration)
        loss = loss_fn(anchor_embed, positive_embed, ref) 

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record information
        loss_accumlation += loss.item()
        loss_avg = loss_accumlation / (step + 1)
        s = 'train ===> epoch:{} ---- step:{} ----lr:{} ---- loss:{:.4f} ---- loss_avg:{:.4f}'.format(
            epoch, step, current_lr, loss, loss_avg
        )
        pbar_train.set_description(s)
        logger.info(s)

    return loss_avg


def validate(model, val_loader, loss_fn, device, logger, pitch_only=True):
    """
    Validate the model on the validation set.

    Args:
        model: The embedding model.
        val_loader: DataLoader for the validation set.
        loss_fn: Loss function to use.
        device: Device to run the computation on.
        logger: Logger to record the loss.
        pitch_only: Whether to use pitch-only data.

    Returns:
        loss_avg: Average loss over the validation set.
    """
    model.eval()
    val_nb = len(val_loader)
    loss_accumlation = loss_avg = 0
    pbar_val = tqdm(val_loader, total=val_nb, desc='Val')
    
    with torch.no_grad():
        for step, data in enumerate(pbar_val):
            # Move data to device
            if pitch_only:
                anchor_pitch, positive_pitch, ref = data
            else:
                anchor_pitch, positive_pitch, anchor_duration, positive_duration, ref = data
                anchor_duration = anchor_duration.to(device)
                positive_duration = positive_duration.to(device)
            
            anchor_pitch = anchor_pitch.to(device)
            positive_pitch = positive_pitch.to(device)
            ref = ref.to(device) 

            # Forward pass
            anchor_embed = model(anchor_pitch) if pitch_only else model(anchor_pitch, anchor_duration)
            positive_embed = model(positive_pitch) if pitch_only else model(positive_pitch, positive_duration)
            loss = loss_fn(anchor_embed, positive_embed, ref)

            # Record information
            loss_accumlation += loss.item()
            loss_avg = loss_accumlation / (step + 1)
            s = 'val ===> epoch:{} ---- step:{} ---- loss:{:.4f} ---- loss_avg:{:.4f}'.format(
                epoch, step, loss, loss_avg
            )
            pbar_val.set_description(s)
            logger.info(s)

    return loss_avg


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training (e.g., cuda:0)')
    parser.add_argument('--ckpt_file_path', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint/a_2', help='Directory to save the checkpoint files')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--data_source', type=str, choices=['P', 'S', 'C', 'R'], default='P', help='Source of the data. (P: Probability-based, S: Statistical-based, C: Completely random, R: Real data)')
    parser.add_argument('--fake_num', type=int, default=2000000, help='Number of fake data')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path to the training data')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path to the validation data')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--last_lr', type=float, default=1e-5, help='Final learning rate')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--loss_mode', type=str, choices=['combine', 'infonce', 'sim'], default='infonce', help='Loss function mode')
    parser.add_argument('--pitch_only', type=int, default=1, help='Use pitch only model')
    parser.add_argument('--frame_dist_path', type=str, default=None, help='Path to the frame distribution')
    parser.add_argument('--train_data_format', type=str, choices=['sequence', 'matrix'], default='matrix', help='Data format for training')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    device = opt.device
    ckpt_file_path = opt.ckpt_file_path
    ckpt_dir = opt.ckpt_dir
    batch_size = opt.batch_size
    pitch_only = bool(opt.pitch_only)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # else:
    #     print("Checkpoint directory already exists!")
    #     exit()

    # Load dataset
    train_dataset = MidiEmbeddingDataset(data_path=opt.train_data_path, data_source=opt.data_source, fake_num=opt.fake_num, pitch_only=pitch_only, frame_dist_file=opt.frame_dist_path, data_format=opt.train_data_format)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    if opt.val_data_path:
        val_dataset = MidiEmbeddingDataset(data_path=opt.val_data_path, data_source='R', pitch_only=pitch_only)
        val_loader = DataLoader(val_dataset, batch_size)

    # Initialize model, loss function, logger and optimizer
    model = MidiResNet().to(device)
    loss_fn = ControlledMusicEmbeddingLoss(loss_mode=opt.loss_mode).to(device)
    logger = get_logger(opt.ckpt_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=opt.num_epochs * len(train_loader), eta_min=opt.last_lr)

    # Load checkpoint
    if ckpt_file_path is not None:
        checkpoint = torch.load(ckpt_file_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print("Loaded epoch {} successfully!".format(start_epoch))
    else:
        start_epoch = 0
        print("Train the model from scratch without saving previous model!")

    # Create and save config file (yaml)
    config = {
        'device': device,
        'ckpt_dir': ckpt_dir,
        'batch_size': batch_size,
        'data_source': opt.data_source,
        'fake_num': opt.fake_num,
        'train_data_path': opt.train_data_path,
        'val_data_path': opt.val_data_path,
        'lr': opt.lr,
        'last_lr': opt.last_lr,
        'num_epochs': opt.num_epochs,
        'loss_mode': opt.loss_mode,
        'pitch_only': pitch_only,
        'frame_dist_path': opt.frame_dist_path,
        'train_data_format': opt.train_data_format
    }
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)  

    # Training loop
    num_epochs = opt.num_epochs
    save_interval = 1
    for epoch in range(start_epoch + 1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, scheduler, loss_fn, device, logger, pitch_only)
        if opt.val_data_path:
            val_loss = validate(model, val_loader, loss_fn, device, logger, pitch_only)

        if epoch % save_interval == 0:
            # save model
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(ckpt_dir, f'mem_{epoch}.pth').format(epoch))
            print("Model saved at epoch {}".format(epoch))
