from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def get_model(model_name, pretrained=True, num_classes=100):
    if model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "deit":
        model = timm.create_model("deit_base_patch16_224", pretrained=True, num_classes=num_classes)
    elif model_name == "swin":
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)
    elif model_name == "resnet":
        model = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")

    return model

def get_CIFAR_loaders(train_batch_size, test_batch_size, image_size = 224):
    transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def get_ImageNet1K_loaders(train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageNet(root="./data", split="train", download=True, transform=transform)
    test_dataset = datasets.ImageNet(root="./data", split="val", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

class Logger:
    def __init__(self, file_path):
        self.file_path = file_path

    def __call__(self, message):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")
            print(message)