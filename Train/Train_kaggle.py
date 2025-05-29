import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from traine import train
import Build_model as Build

torch.random.manual_seed(1)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {

        'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std)
                ]),
        'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((224, 224), antialias=None),
                    transforms.Normalize(mean, std)
                ])
    }
train_dataset = datasets.ImageFolder(
                root='/tmp/pycharm_project_296/datasets/Training',

                transform=data_transform['train'])

val_dataset = datasets.ImageFolder(
                root='/tmp/pycharm_project_296/datasets/Testing',

                transform=data_transform['val'])
batch_size = 32

train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,

                num_workers=16,

                pin_memory=True)

val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=16,
                pin_memory=True)
if __name__ == '__main__':
    dataset = 'Testing'
    swin_type = 'tiny'
    reg_type, reg_lambda = 'l2', 1e-8
    device = torch.device('cuda')
    epochs = 100
    show_per = 200
    ltoken_num, ltoken_dims = 49, 256
    lf = 4

    model = Build.BuildSparseSwin(
        image_resolution=224,
        swin_type=swin_type,
        num_classes=4,
        ltoken_num=ltoken_num,
        ltoken_dims=ltoken_dims,
        num_heads=16,
        qkv_bias=True,
        lf=lf,
        attn_drop_prob=.0,
        lin_drop_prob=.0,
        freeze_12=False,
        device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5,
                                  weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()


    train(
        train_loader,
        swin_type,
        dataset,
        epochs,
        model,
        lf,
        ltoken_num,
        optimizer,
        criterion,
        device,
        show_per=show_per,
        reg_type=reg_type,
        reg_lambda=reg_lambda,
        validation=val_loader)