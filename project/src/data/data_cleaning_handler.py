import pandas as pd

class DataCleaningHandler :
    def display_reflexion(df : pd.DataFrame):
        print(f"We first quickly check the data : {df.head(5)}")
        print(f"We search for missing data : {df.isnull().sum()}") 
        print(f"And finally check the duplicates : {df.duplicated().sum}")
        input("\nPress Enter to quit...")