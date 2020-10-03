import torch
import torch.nn as nn

import torchvision

import tqdm
import time

from model import VisionTransformer

def select_dataset(task):
    if task == 'mnist':
        print("> Using MNIST dataset")
        train_dataset = torchvision.datasets.MNIST('data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
        test_dataset = torchvision.datasets.MNIST('data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
    else:
        print(f"! Unknown task '{task}'!")
        exit()

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=None, shuffle=True, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=None, shuffle=False, batch_size=BATCH_SIZE)

    return train_loader, test_loader

def create_model(task):
    print("> Initialising Vision Transformer")
    if task == 'mnist':
        img_dim = 28
        patch_dim = 7
        out_dim = 10
        nb_channels = 1
        emd_dim = 64
        nb_heads = 4
        nb_layers = 4
        h_dim = 128
    else:
        print(f"! Unknown task '{task}'!")
        exit()

    return VisionTransformer(
        img_dim, patch_dim, out_dim, nb_channels,
        emd_dim, nb_heads, nb_layers, h_dim,
        dropout=DROPOUT
    )

def train(model, loader, optim, crit, device):
    cumulative_loss = 0.0

    model.train()

    for x, y in tqdm.tqdm(loader):
        optim.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = crit(pred, y)
        cumulative_loss += loss.item()
        loss.backward()
        optim.step()

    return cumulative_loss / len(loader)

def evaluate(model, loader, optim, crit, device):
    cumulative_loss = 0.0
    correct_pred = 0

    model.eval()

    for x, y in loader:
        optim.zero_grad()
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = crit(pred, y)
        cumulative_loss += loss.item()

        _, pred = torch.max(pred.data, -1)
        correct_pred += (pred == y).sum().item()

    return cumulative_loss / len(loader), correct_pred / len(loader)

if __name__ == '__main__':
    TRY_CUDA = True
    DATASET = 'mnist'
    BATCH_SIZE = 64
    NB_EPOCHS = 100
    ALPHA = 3e-4
    DROPOUT = 0.1

    device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
    print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'})")

    train_loader, test_loader = select_dataset(DATASET)
    model = create_model(DATASET).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=ALPHA)
    crit = nn.NLLLoss()
    
    for ei in range(NB_EPOCHS):
        print(f"> Epoch {ei+1}/{NB_EPOCHS}")
        train_loss = train(model, train_loader, optim, crit, device)
        eval_loss, eval_accuracy = evaluate(model, test_loader, optim, crit, device)
        print(f"> Training Loss: {train_loss}")
        print(f"> Evaluation Loss: {eval_loss}")
        print(f"> Evaluation Accuracy: {eval_accuracy:.2f}%\n")

