import torch
from torchvision.transforms import transforms
import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.data.components.preprocessing import *
from transformers import BertTokenizer

class HybridMovieGenreDatasetv2(Dataset):
    def __init__(
            self, 
            set, 
            data_file, 
            user_file,
            rating_file,
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
        else:
            df = process_dat_file(dat_file=data_file, folder_img_path=folder_img_path)
            if set == 'train':
                # df = df.sample(frac=0.8, random_state=1012)
                df = random_oversampling(df, 600)
                df = upsampling_all(df, 2)
            elif set == 'val':
                train_df = df.sample(frac=0.8, random_state=1012)
                df = df.drop(train_df.index)
        
        # user & ratings
        users = pd.read_csv(user_file, sep='::',
                        engine='python',
                        names=['userid', 'gender', 'age', 'occupation', 'zip']).set_index('userid')

        ratings = pd.read_csv(rating_file, engine='python',
                          sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
        
        user_rating = ratings.merge(df[["id", "genre"]], how='inner', on='movieid').merge(users, how='inner', on='userid')
        

        # merge data
        df['num_ratings'] = user_rating.groupby('movieid')['rating'].count()
        
        df['avg_rating'] = user_rating.groupby('movieid')['rating'].sum()
        df['avg_rating'] = df['avg_rating'] / df['num_ratings']
        df['avg_rating'] = (df['avg_rating'] - 0.0) / 5.0

        df['num_female_ratings'] = user_rating[user_rating['gender'] == 'F'].groupby('movieid')['rating'].count()
        df['num_male_ratings'] = user_rating[user_rating['gender'] == 'M'].groupby('movieid')['rating'].count()

        for age_value in [1, 18, 25, 35, 45, 50, 56]:
            column_name = f'num_age_{age_value}_ratings'
            df[column_name] = user_rating[user_rating['age'] == age_value].groupby('movieid')['rating'].count()
            df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())

        for occupation in range(1, 21):
            column_name = f'num_occupation_{occupation}_ratings'
            df[column_name] = user_rating[user_rating['occupation'] == occupation].groupby('movieid')['rating'].count()
            df[column_name] = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
        
        # UserRatingEmbedding
        df['num_ratings'] = (df['num_ratings'] - df['num_ratings'].min()) / (df['num_ratings'].max() - df['num_ratings'].min())
        df['num_female_ratings'] = (df['num_ratings'] - df['num_ratings'].min()) / (df['num_ratings'].max() - df['num_ratings'].min())
        df['num_male_ratings'] = (df['num_ratings'] - df['num_ratings'].min()) / (df['num_ratings'].max() - df['num_ratings'].min())
        
        df.fillna(0, inplace=True)
        self.data = df
        
        # pre-processing title
        self.data['cleaned_title'] = [tokenize(x) for x in self.data.title]
        
        bert_tokens = []
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for title in self.data['cleaned_title']:
            bert_tokens.append(['[CLS]'] + self.bert_tokenizer.tokenize(title) + ['[SEP]'])
            
        input_ids, attn_mask = [], []
        for tokens in bert_tokens:
            input_ids.append(self.bert_tokenizer.convert_tokens_to_ids(tokens))
            attn_mask.append(len(tokens)*[1])
            
        self.max_len = -1
        for index in range(len(input_ids)):
            self.max_len = max(self.max_len, len(input_ids[index]))
            
        padded_input_ids_list, padded_attn_mask_list = [], []
        for i in range(len(input_ids)):
            padded_input_ids = input_ids[i] + [0]*(self.max_len-len(input_ids[i]))
            padded_attn_mask = attn_mask[i] + [0]*(self.max_len-len(attn_mask[i]))
            padded_input_ids_list.append(padded_input_ids)
            padded_attn_mask_list.append(padded_attn_mask)
            
        self.data['padded_input_ids'], self.data['padded_attn_mask'] = padded_input_ids_list, padded_attn_mask_list
        
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
        title_input_ids = self.data.iloc[index].padded_input_ids
        title_attn_mask = self.data.iloc[index].padded_attn_mask
        img_path = self.data.iloc[index].img_path
        genre = self.data.iloc[index].genre
        
        # process user & rating
        row_info = self.data.iloc[index].to_dict()
        
        ratings_general = [row_info[k] for k in ['avg_rating', 'num_ratings', 'num_female_ratings', 'num_male_ratings']]
        ratings_age = [row_info[k] for k in [f'num_age_{age_value}_ratings' for age_value in [1, 18, 25, 35, 45, 50, 56]]]
        ratings_occupation = [row_info[k] for k in [f'num_occupation_{occupation}_ratings' for occupation in range(1, 21)]]
        
        ratings_general = torch.tensor(ratings_general)
        ratings_age = torch.tensor(ratings_age)
        ratings_occupation = torch.tensor(ratings_occupation)
        
        # process title
        title_input_ids = torch.from_numpy(np.array(title_input_ids)).long()
        title_attn_mask = torch.from_numpy(np.array(title_attn_mask)).long()
        
        # process img
        if not os.path.exists(img_path):
            img_tensor = torch.zeros((3, 224, 224))
        else:
            assert os.path.exists(img_path)
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)

        # process label
        genre_vector = np.zeros(len(self.genre2idx))
        for g in genre:
            if g not in self.genre2idx.keys():
                genre_vector[-1] = 1 # others
            else:
                genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()

        return (ratings_general, ratings_age, ratings_occupation), title_input_ids, title_attn_mask, img_tensor, genre_tensor

