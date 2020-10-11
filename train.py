import torch
import torch.nn as nn

import torchvision

import tqdm
import time

from model import VisionTransformer, BaseModel

# Function that returns train and test set dataloaders based on the task at hand
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
    elif task == 'fashion':
        print("> Using FashionMNIST dataset")
        train_dataset = torchvision.datasets.FashionMNIST('data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
        test_dataset = torchvision.datasets.FashionMNIST('data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
    elif task == 'kmnist':
        print("> Using Kuzushiji-MNIST dataset")
        train_dataset = torchvision.datasets.KMNIST('data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
        test_dataset = torchvision.datasets.KMNIST('data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
    elif task == 'cifar10':
        print("> Using CIFAR10 dataset")
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.ToTensor()
            ]
        ))
        test_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]
        ))
    elif task == 'cifar100':
        print("> Using CIFAR100 dataset")
        train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(0.5),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.ToTensor()
            ]
        ))
        test_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
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

# Initialises the VisionTransformer based on the task at hand
def create_model(task):
    print("> Initialising Vision Transformer")
    if task in ['mnist', 'fashion', 'kmnist']:
        img_dim = 28
        patch_dim = 4
        out_dim = 10
        nb_channels = 1
        emd_dim = 64
        nb_heads = 4
        nb_layers = 4
        h_dim = 256
    elif task in ['cifar10']:
        img_dim = 32
        patch_dim = 4
        out_dim = 10
        nb_channels = 3
        emd_dim = 128
        nb_heads = 8
        nb_layers = 2
        h_dim = 256
    elif task in ['cifar100']:
        img_dim = 32
        patch_dim = 4
        out_dim = 100
        nb_channels = 3
        emd_dim = 256
        nb_heads = 8
        nb_layers = 2
        h_dim = 1024
    else:
        print(f"! Unknown task '{task}'!")
        exit()

    return VisionTransformer(
        img_dim, patch_dim, out_dim, nb_channels,
        emd_dim, nb_heads, nb_layers, h_dim,
        dropout=DROPOUT
    )

def create_baseline_model(task):
    if task in ['mnist', 'kmnist', 'fashion']:
        img_dim = 28
        nb_channels = 1
        out_dim = 10
        res_channels = 16
        nb_res_blocks = 5
        mlp_dim = 32
    elif task in ['cifar10']:
        img_dim = 32
        nb_channels = 3
        out_dim = 10
        res_channels = 32
        nb_res_blocks = 10
        mlp_dim = 32
    else:
        print(f"! Unknown task '{task}'!")
        exit()
    
    return BaseModel(img_dim, nb_channels, out_dim, res_channels, nb_res_blocks, mlp_dim)

# Trains the given model on the data loaded by loader for one epoch, given an optimizer and criterion
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

# Evaluates the given model by loss and accuracy given a test dataloader, optimizer(?) and criterion
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

    return cumulative_loss / len(loader), correct_pred * 100.0 / (len(loader) * BATCH_SIZE)

if __name__ == '__main__':
    TRY_CUDA = True
    DATASET = 'mnist'
    BATCH_SIZE = 128
    NB_EPOCHS = 100
    ALPHA = 3e-3 # lr 
    DROPOUT = 0.1
    BASELINE = True

    device = torch.device('cuda:0' if TRY_CUDA and torch.cuda.is_available() else 'cpu')
    print(f"> Device: {device} ({'CUDA is enabled' if TRY_CUDA and torch.cuda.is_available() else 'CUDA not available'})")

    train_loader, test_loader = select_dataset(DATASET)
    if BASELINE:
        model = create_baseline_model(DATASET).to(device)
    else:
        model = create_model(DATASET).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=ALPHA)
    crit = nn.CrossEntropyLoss()
    
    print(f"> Model Summary: ")
    print(model, '\n')

    for ei in range(NB_EPOCHS):
        print(f"> Epoch {ei+1}/{NB_EPOCHS}")
        train_loss = train(model, train_loader, optim, crit, device)
        eval_loss, eval_accuracy = evaluate(model, test_loader, optim, crit, device)
        print(f"> Training Loss: {train_loss}")
        print(f"> Evaluation Loss: {eval_loss}")
        print(f"> Evaluation Accuracy: {eval_accuracy:.2f}%\n")
