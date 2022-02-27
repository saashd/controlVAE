import os
import zipfile
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Root directory for the dataset
data_root = 'data/celeba'
# Path to folder with the dataset
dataset_folder = f'{data_root}/img_align_celeba'

# Path to download the dataset to
download_path = f'{data_root}/img_align_celeba.zip'

# with zipfile.ZipFile(download_path, 'r') as ziphandler:
#     ziphandler.extractall(dataset_folder)


# Create a custom Dataset class
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = natsorted(image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img


def load_celeba(batch_size):
    # Load the dataset
    # Path to directory with all the images
    img_folder = f'{dataset_folder}/img_align_celeba'
    # Spatial size of training images, images are resized to this size.
    image_size = 64
    # Transformations to be applied to each individual image sample
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    # Load the dataset from file and apply transformations
    celeba_dataset = CelebADataset(img_folder, transform)

    ## Create a dataloader

    train_set, test_set = torch.utils.data.random_split(celeba_dataset, [192599, 10000])
    # train_set, test_set = torch.utils.data.random_split(celeba_dataset, [10000, 192599])
    celeba_train = torch.utils.data.DataLoader(train_set,

                                               batch_size=batch_size,
                                               shuffle=True)
    celeba_test = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              shuffle=True)
    return celeba_train, celeba_test


def load_cifar10(batch_size):

    # Transformations to be applied to each individual image sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    return trainset, testloader
