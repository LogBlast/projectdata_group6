import pandas as pd

class DataPreprocessingHandler:
    def preprocess_float(df : pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=['float']).columns:
            df[column] = df[column].round().astype('int')
        return df
    
    def preprocess_categorical_data(df : pd.DataFrame) -> pd.DataFrame:
        numerical_columns = df.select_dtypes(include=['number']).columns.to_list()
        categorical_columns = df.select_dtypes(include=['object']).columns.to_list()

        df_dummies = pd.get_dummies(data=df[categorical_columns], dtype=int)
        df = pd.concat([df[numerical_columns], df_dummies], axis=1) # to keep (or not) the original values that were hot encoded
        return df
    
    def preprocess_dataset(df : pd.DataFrame) -> pd.DataFrame:
        df = DataPreprocessingHandler.preprocess_float(df=df)
        df = DataPreprocessingHandler.preprocess_categorical_data(df=df)
        return df

    def display_reflexion(df : pd.DataFrame):
        print("We first round float data.")
        print("Then we convert categorical datas into numerical ones through hot encoding")
        input("\nPress Enter to quit...")