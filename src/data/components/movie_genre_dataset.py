import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import os
from PIL import Image
import numpy as np
import pandas as pd

########################################################## PREPROCESSING ##########################################################
def process_dat_file(dat_file, folder_img_path):
    df = pd.read_csv(dat_file, engine='python',sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    df['genre'] = df.genre.str.split('|')
    df['id'] = df.index
    df['img_path'] = df.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg') if os.path.exists(os.path.join(folder_img_path, f'{row.id}.jpg')) else '', axis=1)
    df = df[df['img_path'] != '']
    return df

def random_oversampling(df, target_samples):
    # dataframe
    oversampled_data = df.copy()
    movies_train_copy = df.copy()
    movies_train_copy = movies_train_copy[~movies_train_copy['genre'].apply(lambda x: True if "Drama" in x else False)]
    movies_train_copy = movies_train_copy[~movies_train_copy['genre'].apply(lambda x: True if "Comedy" in x else False)]

    # oversampling
    genres_list = [
        "Film-Noir",
        "Western",
        "Fantasy",
        "Animation",
        "Mystery",
        "Documentary",
        "Musical",
        "War",
        "Crime",
        "Children's",
        "Sci-Fi",
        "Adventure",
        "Horror",
        "Romance",
        "Thriller",
        "Action",
        "Comedy",
        "Drama"
    ]
    for genre in genres_list:
        if genre in ["Drama", "Comedy"]:
            continue
        
        freq = len(oversampled_data[oversampled_data['genre'].apply(lambda x : True if genre in x else False)])
        oversample_factor = target_samples / freq
        if oversample_factor < 1:
            oversample_factor = 1
        genre_data = movies_train_copy[movies_train_copy['genre'].apply(lambda x: True if genre in x else False)]
        oversampled_genre_data = genre_data.sample(int(freq * (oversample_factor - 1)) + 1, replace=True, random_state=1012)
        oversampled_data = pd.concat([oversampled_data, oversampled_genre_data])

    # # print
    # gc = oversampled_data['genre'].explode().value_counts()
    # print("\n__________________ RANDOM OVERSAMPLING ________________")
    # print(f"Number of genres: {len(gc)} \n")
    # print(f"Frequency: \n{gc}")
    # print("\n")
    
    return oversampled_data

def upsampling_all(df, factor):
    upsampling_data = df.copy()
    return pd.concat([upsampling_data] * factor)

########################################################## DATASET ##########################################################

class MovieGenreDataset(Dataset):
    def __init__(
            self, 
            set, 
            data_file, 
            folder_img_path='data/ml1m/content/dataset/ml1m-images', 
            genre_file='data/ml1m/content/dataset/genres.txt'
        ):
        super().__init__()
        self.set = set

        # dataframe
        if set == "ensemble_test":
            df = pd.read_csv(data_file, engine='python',sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
            df['genre'] = df.genre.str.split('|')
            df['id'] = df.index
            df['img_path'] = df.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg') if os.path.exists(os.path.join(folder_img_path, f'{row.id}.jpg')) else '', axis=1)
            self.data = df
        else:
            df = process_dat_file(dat_file=data_file, folder_img_path=folder_img_path)
            if set == 'train':
                df = df.sample(frac=0.8, random_state=1012)
                df = random_oversampling(df, 600)
                df = upsampling_all(df, 2)
            elif set == 'val':
                train_df = df.sample(frac=0.8, random_state=1012)
                df = df.drop(train_df.index)
            self.data = df
        
        # genre to idx
        with open(genre_file, 'r') as f:
            genre_all = f.readlines()
            genre_all = [x.replace('\n','') for x in genre_all]
        self.genre2idx = {genre:idx for idx, genre in enumerate(genre_all)}

        # transform
        if set == 'train':
            self.transform = transforms.Compose([
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.8),
                transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.7),
                transforms.RandomApply([transforms.RandomResizedCrop(224)], p=0.8),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre
        
        # preprocess img
        if self.set == "ensemble_test" and not os.path.exists(img_path):
            img_tensor = torch.zeros((3, 224, 224))
        else:
            assert os.path.exists(img_path)
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)

        # preprocess label
        genre_vector = np.zeros(len(self.genre2idx))
        for g in genre:
            if g not in self.genre2idx.keys():
                genre_vector[-1] = 1 # others
            else:
                genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        return img_tensor, genre_tensor

