import os
import zipfile
import random
import numpy as np
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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


def load_celeba(batch_size, root='../../data/CelebA'):
    # Root directory for the dataset
    data_root = root
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/CelebA'

    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'
    target_location = r'data_root'

    with zipfile.ZipFile(download_path) as zip_file:
        for member in zip_file.namelist():
            if os.path.exists(dataset_folder + r'/' + member) or os.path.isfile(dataset_folder + r'/' + member):
                break
            else:
                zip_file.extract(member, dataset_folder)
    # Load the dataset
    # Path to directory with all the images
    img_folder = f'{dataset_folder}/img_align_celeba'
    # Spatial size of training images, images are resized to this size.

    image_size = 128
    # Transformations to be applied to each individual image sample

    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    # Load the dataset from file and apply transformations
    celeba_dataset = CelebADataset(img_folder, transform)
    train_size = 10000
    # Create a dataloader
    train_set, test_set = torch.utils.data.random_split(celeba_dataset, [train_size, len(celeba_dataset) - train_size])
    data_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=True,
                                              drop_last=True)
    return data_loader


def load_mnist(batch_size, image_size=128):
    # Transformations to be applied to each individual image sample
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST('./data',
                                         transform=transform,
                                         download=True)
    train_size = 10000
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    # Create a dataloader
    data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True)
    return data_loader


class CustomTensorDataset(Dataset):
    def __init__(self, root, size):
        self.size = size
        dataset_zip = np.load(root, encoding='latin1',
                              allow_pickle=True)
        idx = random.sample(range(0, len(dataset_zip['imgs'])), self.size)
        self.imgs = dataset_zip['imgs'][idx]
        self.latents_values = dataset_zip['latents_values'][idx]
        self.latents_classes = dataset_zip['latents_classes'][idx]

    def __getitem__(self, index):
        imgs = torch.tensor(self.imgs[index]).unsqueeze(0).float()
        latents_values = torch.tensor(self.latents_values[index]).unsqueeze(0)
        return imgs, latents_values

    def __len__(self):
        return self.size


def load_dSprites(batch_size=64, root='../data/dSprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'):
    # Load dataset
    ds = CustomTensorDataset(root=root, size=20000)
    data_loader = torch.utils.data.DataLoader(ds,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              shuffle=True,
                                              drop_last=True)
    return data_loader


def main():
    train_data = load_dSprites(15000,'./data/dSprites/dsprites_subset.npz')

if __name__ == '__main__':
    main()