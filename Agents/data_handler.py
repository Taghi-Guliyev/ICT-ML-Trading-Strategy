# data_handler.py
# Purpose: To handle all data ingestion and cleaning, providing a standardized
#          DataFrame for all other modules in the trading system.

import pandas as pd
import config # Import config to use file paths for the example

class DataHandler:
    """
    A modular agent responsible for loading, cleaning, and preparing market data
    from various sources like CSV files or APIs.
    """
    def __init__(self, config):
        """
        Initializes the DataHandler with configuration.

        Args:
            config (dict): A dictionary containing data source parameters.
        """
        self.config = config
        print(f"DataHandler initialized for source: {self.config.get('source', 'N/A')}")

    def _load_from_csv(self):
        """
        Private method to load and process data from a local CSV file.
        """
        file_path = self.config.get('file_path')
        should_slice = self.config.get('slice_data', False)
        if not file_path:
            raise ValueError("Configuration must include 'file_path' for CSV source.")

        try:
            raw = pd.read_csv(file_path, sep='\t')
        except FileNotFoundError:
            print(f"Error: Data file not found at '{file_path}'.")
            return None
        if should_slice:
            print(f"Original file has {len(raw)} rows. Slicing to select the second half.")
            midpoint = len(raw) // 2
            raw = raw.iloc[midpoint:].copy()
            raw.reset_index(drop=True, inplace=True)
        else:
            print(f"Loading all {len(raw)} rows from the file.")

        raw.rename(columns={'<OPEN>':'Open', '<HIGH>':'High', '<LOW>':'Low', '<CLOSE>':'Close'}, inplace=True)
        
        if '<DATE>' in raw.columns and '<TIME>' in raw.columns:
            raw['datetime'] = pd.to_datetime(raw['<DATE>'] + ' ' + raw['<TIME>'])
            raw.set_index('datetime', inplace=True)
        else:
            raise ValueError("CSV file must contain '<DATE>' and '<TIME>' columns.")
            
        df = raw[['Open','High','Low','Close']].copy().astype(float)
        print(f"Successfully loaded and cleaned {len(df)} rows from {file_path}")
        return df

    def _load_from_api(self):
        """
        Placeholder for a future API connection.
        """
        raise NotImplementedError("API data loading is not yet implemented.")

    def get_data(self):
        """
        Public method to fetch data based on the configured source.
        """
        source_type = self.config.get('source')
        if source_type == 'csv':
            return self._load_from_csv()
        elif source_type == 'api':
            return self._load_from_api()
        else:
            raise ValueError(f"Unknown data source specified in config: '{source_type}'")

# --- Main Execution Block ---
def main():
    """
    Main function to run an example of the DataHandler's functionality.
    This block will only run when data_handler.py is executed directly.
    """
    print("--- Running DataHandler Example ---")
    
    # Example for loading training data
    config_training = {'source': 'csv', 'file_path': config.TRAINING_DATA_FILE}
    training_data_agent = DataHandler(config_training)
    df_train = training_data_agent.get_data()
    
    if df_train is not None:
        print("\n--- Training Data Sample ---")
        print(df_train.head())
        print("--------------------------\n")
    
    # Example for loading backtesting data
    config_backtesting = {'source': 'csv', 'file_path': config.BACKTEST_DATA_FILE}
    backtesting_data_agent = DataHandler(config_backtesting)
    df_backtest = backtesting_data_agent.get_data()

    if df_backtest is not None:
        print("\n--- Backtesting Data Sample ---")
        print(df_backtest.head())
        print("-----------------------------\n")

if __name__ == "__main__":
    main()