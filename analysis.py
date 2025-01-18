import random

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class MyDataSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.X[index]
        _y = self.Y[index]

        return _x, _y

class HybridModel(nn.Module):
    def __init__(self, cnn_channels=256, cnn_kernel=8, lstm_hidden=50, dropout=0.279, layers_cnn=2):
        super(HybridModel, self).__init__()
        
        # CNN-LSTM
        self.cnn_lstm = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1 if i==0 else cnn_channels, cnn_channels, cnn_kernel),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool1d(1),
                nn.Dropout(dropout)
            ) for i in range(layers_cnn)
        ])
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1])
        for cnn in self.cnn_lstm:
            x = cnn(x)
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        x = h[-1]
        x = self.mlp(x)
        return x
    
def Create_Model_Input(dataset = 'kuo_fepb', simple_onehot = False):
    df = pd.read_csv("compiled_sequence_to_expression.csv.gz", compression='gzip')
    df = df[df["experiment"] == dataset]

    # remove rows where "mean_fluo" is NaN or zero
    print("filtering:", df.shape)
    df = df[df["mean_fluo"].notna()]
    df = df[df["mean_fluo"] != 0]
    print("filtering:", df.shape)

    # trim anywhitespace from the UTR column
    df["UTR"] = df["UTR"].str.strip()

    # split the UTR field into individual columns and assign to a new dataframe
    df_utr = df["UTR"].str.split("", expand=True)
    df_utr = df_utr.drop(columns=[0, 31])
    print(df_utr.shape)
    
    if simple_onehot:
        # get the number of unique characters in each column
        total_unique = 0
        for i in range(1, 31):
            total_unique += df_utr[i].nunique()
        print("total unique bases:", total_unique)

        # one hot encode the data
        #df_utr_onehot = pd.get_dummies(df_utr)

        enc = OneHotEncoder()
        df_utr_onehot = enc.fit_transform(df_utr).toarray()

        # convert to integer
        df_utr_onehot = df_utr_onehot.astype(int)

        # write the one hot encoded data to a file using pandas
        # convert to a dataframe
        df_utr_onehot = pd.DataFrame(df_utr_onehot)

    else:
        num_samples = len(df_utr)
        one_hot_array = np.zeros((num_samples, 30 * 4))

        # Define the bases and their positions in each 4-column block
        bases = {'A': 0, 'G': 1, 'C': 2, 'U': 3}

        # For each position in the sequence
        for pos in range(1, 31):
            col = df_utr[pos]
            start_idx = (pos-1) * 4
            
            # For each base in this position
            for base, offset in bases.items():
                one_hot_array[:, start_idx + offset] = (col == base).astype(int)

        # Convert to dataframe with meaningful column names
        columns = []
        for pos in range(1, 31):
            for base in ['A', 'G', 'C', 'U']:
                columns.append(f'pos_{pos}_{base}')
        df_utr_onehot = pd.DataFrame(one_hot_array, columns=columns)
    
    print(df_utr_onehot.shape)
    
    X = df_utr_onehot.to_numpy()
    Y = df["mean_fluo"].values
    
    test_size = 0.2  # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=True)
    
    return  X_train, X_test, y_train, y_test