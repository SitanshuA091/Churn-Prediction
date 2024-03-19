import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
def encode_data(dataframe_series):
        if dataframe_series.dtype=='object':
            dataframe_series = LabelEncoder().fit_transform(dataframe_series)
        return dataframe_series
def preprocess(dataframe):
    dataframe_preprocessed = dataframe.apply(lambda x: encode_data(x))
    return dataframe_preprocessed