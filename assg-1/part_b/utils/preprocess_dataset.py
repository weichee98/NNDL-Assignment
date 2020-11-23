import numpy as np
from utils.seed import SEED
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

class PreprocessDataset:
    
    def __init__(self, df, feature_columns, label_column, test_ratio=0.3, fold=5):
        self.__scaler = MinMaxScaler()
        self.__fold = fold
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = self.__split(df, feature_columns, label_column)
        self.__preprocess()
        
    def __split(self, df, feature_columns, label_column):
        X = df[feature_columns]
        y = df[label_column].to_numpy()
        return train_test_split(X, y, test_size=0.3, random_state=SEED)
    
    def __preprocess(self):
        self.__scaler.fit(self.__X_train)
        self.__X_train = self.__scaler.transform(self.__X_train)
        self.__X_test = self.__scaler.transform(self.__X_test)      
    
    def get_train(self):
        return self.__X_train, self.__y_train
    
    def get_test(self):
        return self.__X_test, self.__y_test
    
    def get_kfold(self):
        return KFold(n_splits=self.__fold).split(self.__X_train) # generator object
    
    def preprocess_input(self, X, y=None):
        if y is None:
            return self.__scaler.transform(X)
        else:
            return self.__scaler.transform(X), y