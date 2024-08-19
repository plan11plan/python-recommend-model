from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random

class MakeCFDataSet():
    """
    GraphDataSet 생성
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, 'tn_visit_area_info_방문지정보_sum2.csv'), low_memory=False)

        # 'TRAVEL_ID'가 'd'로 시작하는 데이터 필터링
        d_starting_data = self.df[self.df['TRAVEL_ID'].str.startswith('d')]

        # 2000개 랜덤 샘플링
        d_starting_sample = d_starting_data.sample(n=2000, random_state=42)

        # 나머지 데이터 필터링
        other_data = self.df[~self.df['TRAVEL_ID'].str.startswith('d')]

        # 최종 데이터프레임 생성
        self.df = pd.concat([d_starting_sample, other_data], ignore_index=True)
        # 인덱스 초기화
        self.df.reset_index(drop=True, inplace=True)

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder('POI_ID')
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder('TRAVEL_ID')
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['POI_ID'].apply(lambda x : self.item_encoder[x])
        self.df['user_idx'] = self.df['TRAVEL_ID'].apply(lambda x : self.user_encoder[x])

        self.exist_users = [i for i in range(self.num_user)]
        self.exist_items = [i for i in range(self.num_item)]
        self.user_train, self.user_valid = self.generate_sequence_data()

    def generate_encoder_decoder(self, col : str) -> dict:
        """
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item in zip(self.df['user_idx'], self.df['item_idx']):
            users[user].append(item)

        for user in users:
            np.random.seed(self.config.seed)

            user_total = users[user]
            valid = np.random.choice(user_total, size = self.config.valid_samples, replace = False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid # valid_samples 개수 만큼 검증에 활용 (현재 Task와 가장 유사하게)

        return user_train, user_valid

    def neg_sampling(self, users):

        neg_sampling_cnt = 3

        def sample_neg_items_for_u(u, num):
            neg_items = list(set(self.exist_items) - set(self.user_train[u]))
            neg_batch = random.sample(neg_items, num)
            return neg_batch

        _users, neg_items = [], []
        for user in users:
            neg_items += sample_neg_items_for_u(user, neg_sampling_cnt)
            _users += [user] * neg_sampling_cnt

        return _users, neg_items

    def get_train_valid_data(self):
        return self.user_train, self.user_valid
    
    
class CFDataset(Dataset):
    def __init__(self, user_train):
        self.users = []
        self.items = []
        for user in user_train.keys():
            self.items += user_train[user]
            self.users += [user] * len(user_train[user])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]

        return user, item