import os
import zipfile

import numpy as np
import torch
import torchvision
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


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
    # Root directory for the dataset
    data_root = '../data/celeba'
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'

    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'
    target_location = r'data_root'

    with zipfile.ZipFile(download_path) as zip_file:
        for member in zip_file.namelist():
            if os.path.exists(dataset_folder + r'/' + member) or os.path.isfile(dataset_folder + r'/' + member):
                break
            else:
                zip_file.extract(member, dataset_folder)

    # with zipfile.ZipFile(download_path, 'r') as ziphandler:
    #     ziphandler.extractall(dataset_folder)

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

    train_set, test_set = torch.utils.data.random_split(celeba_dataset, [10000, 192599])
    data_loader = torch.utils.data.DataLoader(train_set,

                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader


def load_dsprites(batch_size):
    root = '../data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    if not os.path.exists(root):
        import subprocess
        print('Now download dsprites-dataset')
        subprocess.call(['./download_dsprites.sh'])
        print('Finished')
    data = np.load(root, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
    train_kwargs = {'data_tensor': data}
    dset = CustomTensorDataset

    train_data = dset(**train_kwargs)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True, )

    data_loader = train_loader

    return data_loader


def load_mnist(batch_size):
    data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=batch_size,
        shuffle=True)
    return data_loader
