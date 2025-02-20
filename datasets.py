from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_dataset(dataset):
    transform_dict = {
        "MNIST": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        "CIFAR10": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    dataset_dict = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10
    }

    transform = transform_dict[dataset]
    dataset_class = dataset_dict[dataset]

    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader