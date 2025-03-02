
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
    Represents a dataset of stock market data for use in machine learning models.

    This class loads, processes, and manages historical stock market data for a
    given symbol, either for training or testing purposes, as specified by the
    `is_training` flag. It allows for data preprocessing, including splitting
    datasets, creating input-target pairs, and saving/loading datasets to/from
    disk.

    Attributes:
        symbol (str): The stock symbol for which the dataset is created.
        is_training (bool): Indicates whether the dataset is for training or not.
        input_file (str): File path to save/load the input data tensor.
        target_file (str): File path to save/load the target data tensor.
        short_term_time_period (int): The time period for short-term data processing.
        spilt_incides (int): The number of indices to split the dataset into smaller chunks.
        timezone: The time zone for the dataset, set to UTC.
        timeframe: The time frame of the stock data, defaulted to `mt5.TIMEFRAME_M5`.
        count (int): The number of data points to fetch from the source.
        training_data_len (int): The length of training data, calculated as 80% of
            the total data length.
        inputs: Tensor representation of the input data.
        targets: Tensor representation of the target data.
    """
    def __init__(self, symbol :str, is_training: bool, input_file: str, target_file: str, short_term_time_period: int,
                 spilt_incides: int):

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

            self.create_input_and_target_data(df, is_training, short_term_time_period, spilt_incides)
        else:
            self.inputs = torch.load(input_file, weights_only=True)
            self.targets = torch.load(target_file,  weights_only=True)

    def create_input_and_target_data(self, df: pandas.DataFrame, is_training: bool, short_term_time_period: int,
                                     spilt_incides: int):
        """
        Creates input and target data for training or testing by processing the given dataset (DataFrame). This method
        segments the dataset into appropriate training and testing sets, splits the data into smaller subsets based on the
        provided parameters, and generates formatted label arrays for the model. The processed data is then saved to
        designated files for later use.

        Args:
            df (pandas.DataFrame): The dataset containing numerical data to be processed for inputs and targets.
            is_training (bool): A flag indicating whether to process the data in training mode or testing mode.
            short_term_time_period (int): The time period length used to segment data for short-term predictions.
            spilt_incides (int): The number of segments into which the data is split for label generation.
        """
        labels = np.array([])

        # create training and testing dataframes
        training_data = df[:self.training_data_len]
        testing_data = df[self.training_data_len:]

        training_data_raw = training_data.to_numpy(dtype=np.double)
        testing_data_raw = testing_data.to_numpy(dtype=np.double)

        len_training_data = len(training_data_raw)

        for raw_data_idx in range(len_training_data - (spilt_incides * 2), 0, -1):

            if raw_data_idx > len_training_data - short_term_time_period - spilt_incides - 1:

                raw_data = training_data_raw[raw_data_idx:]
                # spilt the data into 10 lists
                spilt_data  = np.array_split(raw_data, spilt_incides * 2)

                for data in spilt_data:
                    labels = np.append(labels, data[-1][:4])
            else:
                long_raw_term_data = training_data_raw[raw_data_idx: raw_data_idx + short_term_time_period]
                short_raw_term_data = training_data_raw[raw_data_idx + short_term_time_period:]

                # spilt for short term and long term predict
                short_term_data = np.array_split(short_raw_term_data, spilt_incides)
                long_term_data = np.array_split(long_raw_term_data, spilt_incides)

                # adding data to labels array
                for data in short_term_data:
                    labels = np.append(labels, data[-1][:4])

                for data in long_term_data:
                    labels = np.append(labels, data[-1][:4])

        # reshaping the labels to length of input by spilt_window * 2 * 4
        labels = labels.reshape(-1, spilt_incides * 2 * 4)

        if is_training:
            self.inputs = torch.from_numpy(training_data_raw[: -spilt_incides * 2])
            self.targets = torch.from_numpy(labels)

            torch.save(self.inputs, self.input_file)
            torch.save(self.targets, self.target_file)
        else:
            self.inputs = testing_data_raw[: -spilt_incides * 2]
            self.targets = labels

            torch.save(torch.Tensor(self.inputs), self.input_file)
            torch.save(torch.Tensor(self.targets), self.target_file)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        inputs = self.inputs[idx]
        target = self.targets[idx]

        return inputs, target