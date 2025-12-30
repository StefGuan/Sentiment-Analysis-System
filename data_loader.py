# basic data loader module
import pandas as pd
import numpy as np
from io import StringIO
import os

class DataLoader:
    def __init__(self):
        self.data = None
        self.data_path = 'D:\\Python\\PythonCode\\Data_science\\data\\waimai_10k.csv'

    """Load data from a local CSV file"""
    def load_local(self):
        try:    # check if file exists
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                return self.data
            else:
                print("File not found.")
                return None
        except Exception as e:
            print(f"Something went wrong: {e}")
            return None

    """Get basic information about the dataset"""
    def get_data_info(self):
        if self.data is None:
            print("Data not loaded.")
            return
        
        info = {
            'total_records': len(self.data),
            'columns': list(self.data.columns),
            'positive_count': self.data['label'].sum(),
            'negative_count': len(self.data) - self.data['label'].sum(),
            'positive_ratio': self.data['label'].mean(),
            'missing_values': self.data.isnull().sum().to_dict()
        }

        print("="*50)   # separator line  
        print("Dataset Information:")
        for key, value in info.items():
            if key not in ['missing_values']:
                print(f"{key}: {value}")

        print("Missing Values per Column:")
        for col, count in info['missing_values'].items():
            print(f"  {col}: {count}")

        return info
    
# test the DataLoader class
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_local()
    if data is not None:
        loader.get_data_info()