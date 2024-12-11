from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
from PIL import Image
import torch
import random

def get_ImageNet1K_len(batch_size):
    path = "./data/image_net/train_images"
    assert os.path.exists(path), "ImageNet Not Found"

    images = os.listdir(path)
    return len(images)//batch_size

def get_ImageNet1K_labels():
    path = "./data/image_net/train_images"
    assert os.path.exists(path), "ImageNet Not Found"

    y_map = {}
    images = os.listdir(path)
    for img in images:
        label = img.split("_")[0]

        if label not in y_map.keys():
            y_map[label] = len(y_map.keys())

    return y_map

def get_model_name(model_name):
    if model_name == "vit":
        model_name = "vit_base_patch16_224"
    elif model_name == "deit":
        model_name = "deit_base_patch16_224"
    elif model_name == "swin":
        model_name = "swin_base_patch4_window7_224"
    elif model_name == "resnet":
        model_name = "resnet50"
    else:
        raise ValueError("Invalid model name")

    return model_name

def get_model(model_name, pretrained=True, num_classes=100):
    if model_name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    elif model_name == "deit":
        model = timm.create_model("deit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    elif model_name == "swin":
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    elif model_name == "resnet":
        model = timm.create_model("resnet50", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")

    return model

def get_CIFAR_loaders(train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform2)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def get_ImageNet1K_loaders_train(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    global y_map

    path = "./data/image_net/train_images"
    assert os.path.exists(path), "ImageNet Not Found"

    images = os.listdir(path)
    random.shuffle(images)

    for i in range(0, len(images), batch_size):
        X, X_t = [], images[i: i+batch_size]
        y = []
        for img in X_t:
            try:
                img_label = img.split("_")[0]
                img = Image.open(path+"/"+img)
                img = transform(img)
                X.append(img)
                y.append(y_map[img_label])
            except:
                pass

        X = torch.stack(X)
        y = torch.tensor(y)

        yield X, y


def get_ImageNet1K_loaders_test(batch_size):
    global y_map
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    path = "./data/image_net/val_images"
    assert os.path.exists(path), "ImageNet Not Found"

    images = os.listdir(path)
    random.shuffle(images)

    for i in range(0, len(images), batch_size):
        X, X_t = [], images[i: i+batch_size]
        y = []
        for img in X_t:
            try:
                img_label = img.split("_")[0]
                img = Image.open(path+"/"+img)
                img = transform(img)
                X.append(img)
                y.append(y_map[img_label])
            except:
                pass

        X = torch.stack(X)
        y = torch.tensor(y)

        yield X, y    

class Logger:
    def __init__(self, file_path):
        self.file_path = file_path

    def __call__(self, message):
        with open(self.file_path, "a") as f:
            f.write(f"{message}\n")
            print(message)


# y_map = get_ImageNet1K_labels()

if __name__ == "__main__":
    loader  = get_ImageNet1K_loaders_train(32)

    for batch in loader:
        print(len(batch))
        X,y = batch