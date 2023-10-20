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
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )
    return model


def train(model, writer, coord_inputs, coord_outputs, steps=2001):
    """Train model on dataset."""
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # create loss function
    loss_fn = nn.SmoothL1Loss()
    # loop over epochs
    for step in range(steps):
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
        # print loss
        print(f"Step {step} | Loss {loss.item()}")
        
        # call add_image every 100 steps
        if step % 100 == 0:
            add_image(coord_inputs, y_hat, writer, step)


def add_image(coords, values_rgb, writer, step):
    """Reconstruct the 32x32 rgb image from pixel coords and values.
    
    Then add to summary writer.
    """
    coords = coords.cpu().detach()
    values_rgb = values_rgb.cpu().detach()
    # float, move and scale
    coords *= 15.5
    coords += 15.5
    # round coords to nearest integer
    coords = torch.round(coords).to(torch.int64)
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
    # float, move and scale
    index = index.to(torch.float32)
    index -= 15.5
    index /= 15.5
    
    outputs = img.permute(1, 2, 0).reshape((-1, 3))
    
    train(model.cuda(), writer, index.cuda(), outputs.cuda())

if __name__ == "__main__":
    main()
