from torch.utils.data import DataLoader
from Stock_prediction.datasets.lstmStockDataset import StockDataset as LSTMDataset
from Stock_prediction.datasets.nnStockDataset import StockDataset as NNDataset


def create_dataloader(symbol: str='USDJPY', look_back: int= 32, batch_size: int = 64, model_type: str = "nn",
                      optuna_study = False):
    """
    Creates and returns DataLoader objects for training and testing stock data.

    This function constructs training and testing datasets using the provided stock symbol and look-back period.
    It then initializes PyTorch DataLoader instances for both the training and testing datasets with the specified batch size.

    Args:
        optuna_study:
        batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
        symbol (str): The stock symbol or ticker for which the dataset is to be created.
        look_back (int): The number of previous time steps (look-back period) used as input to predict the next time step.

    Returns:
        tuple: A tuple containing two DataLoader instances:
            - train_dataloader: DataLoader for the training dataset.
            - test_dataloader: DataLoader for the testing dataset.

    Example:
        train_loader, test_loader = create_dataloader(batch_size=32, symbol='USDJPY', look_back=30)
    """
    if model_type == "lstm":
        training_input_file = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\Deep-Learning-Practice"
                               "\\Stock_prediction\\datasets\\training_lstm_input_data.pt")
        training_target_file = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\Deep-Learning-Practice"
                               "\\Stock_prediction\\datasets\\training_lstm_target_data.pt")
        testing_input_file = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\Deep-Learning-Practice"
                               "\\Stock_prediction\\datasets\\testing_lstm_input_data.pt")
        testing_target_file = ("C:\\Users\\Harsh\\Desktop\\Coding Projects\\GitHub\\Deep-Learning-Practice"
                               "\\Stock_prediction\\datasets\\testing_lstm_target_data.pt")

        train_dataset = LSTMDataset(symbol, look_back, True, training_input_file, training_target_file)
        test_dataset = LSTMDataset(symbol, look_back, False, testing_input_file, testing_target_file)
    elif model_type == "nn":
        train_dataset = NNDataset(symbol, True)
        test_dataset = NNDataset(symbol, False)
    else:
        train_dataset = None
        test_dataset = None
        print("Model type not implemented")

    if train_dataset is not None or test_dataset is not None:
        # Create data loaders.
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        return train_dataloader, test_dataloader
    else:
        return None, None
