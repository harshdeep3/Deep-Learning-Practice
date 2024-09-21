

from torchvision import datasets
from torchvision.transforms import ToTensor

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
from Stock_prediction.MT5_Link import get_historic_data
from torch.utils.data import Dataset


class StockDataset(Dataset):
    
    def __init__(self, symbol:str, look_back: int, is_training: bool):
        # metatrader info
        # timeframe objects https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
        self.timeframe = mt5.TIMEFRAME_M5
        self.symbol = symbol
        # set time zone to UTC
        self.timezone = pytz.timezone("Etc/UTC")
        
        # goes back to 1971-08-11
        self.count = 13500

        # get data
        df = get_historic_data(self.symbol, self.timeframe, self.count)
        df = df.set_index('time')
        
        # length of dataset
        self.training_data_len = round(len(df) * 0.8)
        
        self.create_input_and_target_data(df, look_back, is_training)


    def create_input_and_target_data(self, df, look_back, is_training):
        
        # create training and testing dataframes
        training_data = df[:self.training_data_len]     
        testing_data = df[self.training_data_len:]
        
        training_data_raw = training_data.to_numpy()
        testing_data_raw = testing_data.to_numpy()
        
        # inputs
        training_data_input = []
        testing_data_input = []
        
        # targets
        training_data_target = []
        testing_data_target = []
        
        # create inputs and targets
        if is_training:
            
            # create all possible sequences of length look_back
            for index in range(len(training_data_raw) - look_back):
                # input data
                training_data_input.append(training_data_raw[index: index + look_back])
                # target data
                training_data_target.append(training_data_raw[index + look_back])

            self.inputs = training_data_input
            self.targets = training_data_target
        else:

            # create all possible sequences of length look_back
            for index in range(len(testing_data_raw) - look_back):
                # input data
                testing_data_input.append(testing_data_raw[index: index + look_back])
                # target data
                testing_data_target.append(testing_data_raw[index + look_back])

            self.inputs = training_data_input
            self.targets = training_data_target
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        inputs = self.inputs[idx]
        target = self.target[idx]
        
        return inputs, target