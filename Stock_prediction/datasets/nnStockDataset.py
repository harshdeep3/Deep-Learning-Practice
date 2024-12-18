
import os
import torch
import MetaTrader5 as mt5
import pytz
from Stock_prediction.MT5_Link import get_historic_data
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(self, symbol :str, is_training: bool, input_file: str, target_file: str):
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


    def create_input_and_target_data(self, df, is_training):

        # create training and testing dataframes
        training_data = df[:self.training_data_len]
        testing_data = df[self.training_data_len:]

        training_data_raw = training_data.to_numpy()
        testing_data_raw = testing_data.to_numpy()

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