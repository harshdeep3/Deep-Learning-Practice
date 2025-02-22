import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

LOGIN = 52069703
SERVER = "ICMarketsSC-Demo"
# password
PASSWORD = "8z$y2UX5s6aPFb"

# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ", mt5.__author__)
print("MetaTrader5 package version: ", mt5.__version__)


class MT5Class:
    """
    Handles MetaTrader 5 terminal login and retrieves account information.

    This class manages the login process to the MetaTrader 5 trading terminal using given credentials
    and retrieves account information. It provides methods to access and display this information
    in an organized format, using a pandas DataFrame for better readability.

    Attributes:
        mt5_result (bool): Stores the login operation status. True if login is successful,
            otherwise False.
        account_info (pd.DataFrame): DataFrame storing account information with columns
            'property' and 'value'.
    """
    def __init__(self):
        self.mt5_result = None
        self.account_info = None

    def login_to_metatrader(self):
        """
        Logs in to the MetaTrader 5 terminal using the given account credentials.

        This function initializes a connection to the MetaTrader 5 terminal and attempts to
        log in using the provided login credentials and account server. It checks whether
        the login attempt is successful, and if it fails, it terminates the process with
        an error message.

        Attributes:
            SERVER (str): The server name associated with the trading account.
            LOGIN (int): The numeric ID of the trading account, used for authentication.
            PASSWORD (str): The password for the MetaTrader 5 trading account.
            mt5_result (bool): The status of the login operation. True if successful,
                otherwise False.

        Raises:
            SystemExit: Exits the process if the login attempt is unsuccessful.
        """
        # Connect to the MetaTrader 5 terminal
        mt5.initialize()

        # Log in to the terminal with your account credentials
        account_server = SERVER
        # this needs to be an integer
        login = LOGIN
        password = PASSWORD
        self.mt5_result = mt5.login(login, password, account_server)

        if not self.mt5_result:
            print("Login failed. Check your credentials.")
            quit()

    def get_acc_info(self):
        """
        Retrieves and processes the MetaTrader 5 account information, storing it in a DataFrame
        and printing the structured output.

        This method accesses the account information from MetaTrader 5 and checks if the information
        retrieval is successful. If successful, it converts the account information into a dictionary
        format and stores it in a pandas DataFrame. The DataFrame organizes the data into two columns:
        'property' and 'value'. The resulting DataFrame is then printed to the console. If the account
        information is unavailable, an appropriate message is displayed.

        Returns:
            None
        """
        if mt5.account_info() is None:
            print("Account info is None!")
        else:
            account_info_dict = mt5.account_info()._asdict()
            self.account_info = pd.DataFrame(list(account_info_dict.items()), columns=['property', 'value'])
            print(self.account_info)


def get_historic_data(fx_symbol: str, fx_timeframe, fx_count: int) -> pd.DataFrame | None:
    """
    Fetches historic data for a given financial instrument and timeframe using MetaTrader 5 (MT5) API.

    This function retrieves historical price data for a specified financial symbol and timeframe from
    MetaTrader 5 using the provided count of data points. The data is returned as a pandas DataFrame,
    where the 'time' field is converted to a datetime format. If data cannot be retrieved, an error
    message is printed, and None is returned.

    Args:
        fx_symbol: The financial instrument symbol (e.g., 'EURUSD') to fetch data for.
        fx_timeframe: The timeframe for the data (e.g., 'h1', 'd1') as defined in MT5 timeframes.
        fx_count: The number of historical data points to retrieve.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the historical price data where 'time' is in
        datetime format, or None if data retrieval fails.
    """
    rates = mt5.copy_rates_from_pos(fx_symbol, fx_timeframe, 0, fx_count)
    # dataframe
    historic_df = pd.DataFrame(rates)
    # changing the time to datetime
    if "time" in historic_df.keys():
        historic_df['time'] = pd.to_datetime(historic_df['time'], unit='s')
        return historic_df
    else:
        print("\n\nData not found! Check MT5 connection!")
        return None


if __name__ == "__main__":
    mt5_obj = MT5Class()
    mt5_obj.login_to_metatrader()
    mt5_obj.get_acc_info()

    # timeframe objects https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py
    timeframe = mt5.TIMEFRAME_M5
    symbol = 'USDJPY'
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2010, 1, 10, tzinfo=timezone)
    utc_to = datetime(2020, 1, 11, tzinfo=timezone)
    
    # goes back to 1971-08-11
    count = 13500

    # print account info
    # mt5_obj.get_acc_info()
    
    # get data
    df = get_historic_data(symbol, timeframe, count)
    df = df.set_index('time')

    print(df.head())
    # Disconnect from the terminal
    mt5.shutdown()
