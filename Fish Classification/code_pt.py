import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.io
from torchvision.transforms import Resize, Compose, Lambda,  RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class FishDataset(Dataset):
    def __init__(self, annotation_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(idx)
        img_path = self.img_labels.iloc[idx, 0]
        image = torchvision.io.read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.type(torch.float32)/255
        return image, label


class CustomCNN(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.resize = Resize((size, size), antialias=True)
        self.augment = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(60),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(3)

    def forward(self, X):
        output = self.resize(X)
        if (self.training):  # augment only during training
            output = self.augment(output)
        output = F.relu(self.bn1(self.conv1(output)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = self.flatten(output)
        output = self.fc1(output)
        return output


class VGG_BASED_MODEL(nn.Module):
    def __init__(self, size=128):
        super().__init__()
        self.resize = Resize((size, size), antialias=True)
        self.augment = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(60),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.feature_extractor = nn.Sequential(*list(vgg16(VGG16_Weights.DEFAULT).children())[:-2])  # wihout the fully connected head
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(3)

    def forward(self, X):
        output = self.resize(X)
        if (self.training):
            output = self.augment(output)
        output = self.feature_extractor(output)
        output = self.flatten(output)
        output = self.fc1(output)
        return output


def examples(dataset, label_map):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols*rows+1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, probs = dataset[sample_idx]
        label = torch.argmax(probs).item()
        figure.add_subplot(rows, cols, i)
        plt.title(label_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()


def get_dataset():
    folder = os.path.dirname(os.path.realpath(__file__))
    dataset = FishDataset(f'{folder}/data/annotations.csv',
                          transform=Resize(200, antialias=True),
                          target_transform=Lambda(lambda y: torch.zeros(3, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
    return dataset


def loaders(dataset, batch_size, shuffle, random_seed, train_ratio, validate_ratio):
    # batch_size = 1
    # random_seed = 1
    # shuffle = True
    # train_ratio = 0.6
    # validate_ratio = 0.2
    dataset_length = len(dataset)
    split1 = int(np.round(dataset_length*train_ratio))
    split2 = int(np.round(dataset_length*(train_ratio + validate_ratio)))

    indices = list(range(dataset_length))

    if (shuffle):
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, validate_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
    print(f'Train Subset Ratio Is: {len(train_indices)/dataset_length}, length: {len(train_indices)}')
    print(f'Validate Subset Ratio Is: {len(validate_indices)/dataset_length} length: {len(validate_indices)}')
    print(f'Test Subset Ratio Is: {len(test_indices)/dataset_length} length: {len(test_indices)}')

    errors_train = [1 if (idx in test_indices or idx in validate_indices) else 0 for idx in train_indices]
    errors_validate = [1 if (idx in test_indices or idx in train_indices) else 0 for idx in validate_indices]
    errors_test = [1 if (idx in train_indices or idx in validate_indices) else 0 for idx in test_indices]

    errors = errors_train+errors_validate+errors_test
    if (np.sum(errors)):
        print('ERROR: There are intersections between the train, valdiate and test subsets')
    else:
        print('CONFIRMED: All subsets are unique')

    train_sampler = SubsetRandomSampler(train_indices)
    validate_sampler = SubsetRandomSampler(validate_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validate_loader = DataLoader(dataset, batch_size=batch_size, sampler=validate_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, validate_loader, test_loader


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f'The neural network will use "{device}" device')
    return device


def train_once(loader, model, loss_fun, optimizer, device):
    model.train()
    for (X, y) in loader:
        optimizer.zero_grad()
        X = X.to(device)
        y_true = y.to(device)
        y_pred = model(X)
        loss = loss_fun(y_pred, y_true.argmax(1))
        loss.backward()
        optimizer.step()


def train(model, train_loader, validate_loader, loss_fun, optimizer, device, epoches, print_every):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for e in tqdm(range(epoches), position=0, leave=True):
        train_once(train_loader, model, loss_fun, optimizer, device)
        if (e % print_every) == 0 and print_every != -1:
            train_loss, train_acc = evaluate(train_loader, model, loss_fun, prefix="Train", device=device, verbose=True)
            val_loss, val_acc = evaluate(validate_loader, model, loss_fun, prefix="Validate", device=device, verbose=True)
        else:
            train_loss, train_acc = evaluate(train_loader, model, loss_fun, prefix="Train", device=device, verbose=False)
            val_loss, val_acc = evaluate(validate_loader, model, loss_fun, prefix="Validate", device=device, verbose=False)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)


def evaluate(loader, model, loss_fun, prefix, device, verbose):
    model.eval()
    loss = 0
    true_values = 0
    with torch.no_grad():
        for batch_num, (X, y) in enumerate(loader):
            X = X.to(device)
            y_true = y.to(device)
            y_pred = model(X)

            loss += loss_fun(y_pred, y_true.argmax(1)).item()
            true_values += (y_pred.argmax(1) == y_true.argmax(1)).type(torch.float).sum().item()
        num_batches = len(loader)
        # print(num_batches)
        size = len(loader.sampler.indices)
        # print(size)
        acc = true_values/size
        loss /= num_batches
        if verbose:
            print(f"{prefix}: Loss: {loss:.2f}, Accuracy: {acc*100:.2f}%, Correct: {true_values} out of: {size}")
    return loss, acc


def results(model, loss_fun, train_loader, validate_loader, test_loader, device):
    evaluate(train_loader, model, loss_fun, prefix="Train", device=device, verbose=True)
    evaluate(validate_loader, model, loss_fun, prefix="Validate", device=device, verbose=True)
    evaluate(test_loader, model, loss_fun, prefix="Test", device=device, verbose=True)


def main():
    folder = os.path.dirname(os.path.realpath(__file__))
    dataset = get_dataset()
    label_map = {0: 'Dace', 1: 'Roach', 2: 'Perch'}
    # examples(dataset, label_map)
    batch_size = 256  # change based on the V-RAM available
    random_seed = 5
    shuffle = True
    train_ratio = 0.6
    validate_ratio = 0.2
    train_loader, validate_loader, test_loader = loaders(dataset, batch_size, shuffle, random_seed, train_ratio, validate_ratio)
    device = get_device()
    loss_fun = nn.CrossEntropyLoss()

    # # Custom CNN model
    print("".join(['-']*100))
    eps = 100
    print(f"Training a Custom CNN Model for {eps} epochs")

    model = CustomCNN(size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print_every = 10
    train(model, train_loader, validate_loader, loss_fun, optimizer, device, eps, print_every)
    print("Finished Training \n Results")
    results(model, loss_fun, train_loader, validate_loader, test_loader, device)
    folder = os.path.dirname(os.path.realpath(__file__))
    torch.save(model.state_dict(), f'{folder}/models/custom_weights.pth')
    del model

    # VGG16-based CNN model
    print("".join(['-']*100))
    eps = 100
    print(f"Training a VGG16-based CNN Model for {eps} epochs")

    model = VGG_BASED_MODEL(size=128).to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'The model contains {params} parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    print_every = 10
    train(model, train_loader, validate_loader, loss_fun, optimizer, device, eps, print_every)
    print("Finished Training \n Results")
    results(model, loss_fun, train_loader, validate_loader, test_loader, device)

    torch.save(model.state_dict(), f'{folder}/models/vgg_weights.pth')
    del model


if __name__ == '__main__':
    main()
