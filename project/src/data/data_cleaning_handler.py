import pandas as pd

class DataCleaningHandler :
    def display_reflexion(raw : pd.DataFrame):
        print(raw.head(5))
        print(raw.isnull().sum()) 
        input("\nPress Enter to quit...")