"""Load torch CIFAR dataset and training loop."""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.tensorboard import SummaryWriter


def get_dataset():
    """Get CIFAR10 dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    return dataset


def get_model():
    """Create 4 layer MLP with 3 output neurons."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    return model


def add_positional_encoding(coords: torch.Tensor):
    """Add positional sin cos encoding to [B, 2] coords."""
    # [B, 2]
    coords = coords.clone().to(torch.float32)
    # [B, 2]
    coords -= 15.5
    # [B, 2, 1]
    coords = coords.unsqueeze(-1)
    # [B, 2, 8]
    sin = torch.sin(coords / 16.0 ** (2 * torch.arange(0, 16, 2) / 8))
    # [B, 2, 8]
    cos = torch.cos(coords / 16.0 ** (2 * torch.arange(1, 16, 2) / 8))
    # [B, 2, 16]
    sin_cos = torch.cat([sin, cos], dim=-1)
    # [B, 32]
    sin_cos = sin_cos.flatten(start_dim=1)
    return sin_cos


def train(model, writer, coord_inputs, coord_outputs, coords_orig, steps=30000):
    """Train model on dataset."""
    # count all optimizer steps
    global_step = 0

    epoch_size = 10000
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # create loss function
    loss_fn = nn.SmoothL1Loss()
    # create lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.5, verbose=True
    )
    # loop over epochs
    for epoch in range(1, steps // epoch_size + 1):
        for step in range(1, epoch_size + 1):
            # zero gradients
            optimizer.zero_grad()
            # forward pass
            y_hat = model(coord_inputs)
            # compute loss
            loss = loss_fn(y_hat, coord_outputs)
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()
            global_step += 1
            # print loss
            if step % 100 == 0:
                print(f"Step {global_step} | Loss {loss.item()}")
            if step % 1000 == 0:
                # call add_image for every epoch
                add_image(coords_orig, y_hat, writer, global_step)
            
        # update lr scheduler
        scheduler.step()


def add_image(coords, values_rgb, writer, step):
    """Reconstruct the 32x32 rgb image from pixel coords and values.
    
    Then add to summary writer.
    """
    values_rgb = values_rgb.cpu().detach()
    # make image
    img = torch.zeros((32, 32, 3), dtype=torch.float32)
    img[coords[:, 0], coords[:, 1], :] = values_rgb
    # add to summary writer
    img = img.permute(2, 0, 1)
    writer.add_image("reconstruction", img, step)


def main(example_index: int = 255):
    dataset = get_dataset()
    model = get_model()
    
    # [3, H, W]
    img = dataset[example_index][0]
    # make summary writer
    writer = SummaryWriter("logs")
    # put img into tensorboard summary
    writer.add_image("image", img, 0)
    
    # make index tensor for every pixel in 32x32 image
    x = torch.arange(32)
    y = torch.arange(32)
    x, y = torch.meshgrid(x, y)

    # [32, 32, 2]
    index = torch.stack([x, y], dim=-1)
    # [32*32, 2]
    index = index.reshape((-1, 2))
    # apply positional encoding
    encoded = add_positional_encoding(index)
    
    outputs = img.permute(1, 2, 0).reshape((-1, 3))
    
    train(model.cuda(), writer, encoded.cuda(), outputs.cuda(), index)

if __name__ == "__main__":
    main()
