
import os

import pandas
import torch
import numpy as np
import MetaTrader5 as mt5
import pytz
from Stock_prediction.MT5_Link import get_historic_data
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """
    Represents a dataset for loading and managing historical stock data.

    This class is designed to interact with Metatrader 5 API to fetch historical data
    for a given stock symbol, process it, and save it into a format compatible with
    PyTorch datasets. It supports both training and testing modes, where the dataset is
    split accordingly. Tensors can be loaded from the file if they already exist or
    recreated from the raw data.

    Attributes:
        symbol (str): The stock symbol to fetch the historical data for.
        is_training (bool): Indicates if the dataset is being prepared for training.
        input_file (str): The file path where input tensors are saved or loaded.
        target_file (str): The file path where target tensors are saved or loaded.
        inputs (np.ndarray or torch.Tensor): Input data for the dataset.
        targets (np.ndarray or torch.Tensor): Target data for the dataset.
        timeframe (int): The Metatrader 5 timeframe constant representing the time interval
            of the fetched data.
        timezone (pytz.timezone): The timezone used for aligning timestamps in the dataset
            (set to UTC by default).
        count (int): The number of historical data points retrieved from the API.
        training_data_len (int): The length of the training dataset, determined as 80%
            of the total number of data points.
    """
    def __init__(self, symbol :str, is_training: bool, input_file: str, target_file: str):

        # precision point
        torch.set_printoptions(precision=20)
        np.set_printoptions(precision=20)

        if not os.path.exists(input_file) or not os.path.exists(target_file):
            # metatrader info
            # timeframe objects https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
            self.inputs = None
            self.targets = None
            self.timeframe = mt5.TIMEFRAME_M5
            self.symbol = symbol
            # set time zone to UTC
            self.timezone = pytz.timezone("Etc/UTC")

            self.count = 13500

            # get data
            df = get_historic_data(self.symbol, self.timeframe, self.count)
            df = df.set_index('time')

            # length of dataset
            self.training_data_len = round(len(df) * 0.8)

            self.input_file = input_file
            self.target_file = target_file

            self.create_input_and_target_data(df, is_training)
        else:
            self.inputs = torch.load(input_file, weights_only=True)
            self.targets = torch.load(target_file,  weights_only=True)


    def create_input_and_target_data(self, df: pandas.DataFrame, is_training: bool):
        """
        Creates and initializes input and target datasets for both training and testing data.

        Depending on whether the model is training or not, the method slices the provided
        DataFrame into either training or testing sets. The respective sets are then
        converted to numpy arrays and saved as tensors to the locations specified by
        `self.input_file` and `self.target_file`.

        Args:
            df (pd.DataFrame): The input dataset from which training or testing data
                will be extracted.
            is_training (bool): A flag indicating whether the operation is for training
                data (`True`) or testing data (`False`).
        """
        # create training and testing dataframes
        training_data = df[:self.training_data_len]
        testing_data = df[self.training_data_len:]

        training_data_raw = training_data.to_numpy(dtype=np.double)
        testing_data_raw = testing_data.to_numpy(dtype=np.double)

        if is_training:
            self.inputs = training_data_raw
            self.targets = training_data_raw

            torch.save(torch.Tensor(self.inputs), self.input_file)
            torch.save(torch.Tensor(self.targets), self.target_file)
        else:
            self.inputs = testing_data_raw
            self.targets = testing_data_raw

            torch.save(torch.Tensor(self.inputs), self.input_file)
            torch.save(torch.Tensor(self.targets), self.target_file)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        inputs = self.inputs[idx]
        target = self.targets[idx]

        return inputs, target