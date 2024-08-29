import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ProtoEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, stats_file='encoding_stats.pkl'):
        self.stats_file = stats_file
        self.proto_stats = None
        self.proto_overall_mean = None
    
    def fit(self, X, y=None):
        with open(self.stats_file, 'rb') as f:
            encoding_stats = pickle.load(f)
        
        self.proto_stats, self.proto_overall_mean = encoding_stats['proto']
        return self
    
    def transform(self, X):
        X = X.copy()
        X['proto_target_encoded'] = X['proto'].map(self.proto_stats)
        X['proto_target_encoded'].fillna(self.proto_overall_mean, inplace=True)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X.drop('proto', axis = 1)


class FeatureScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        with open('robust_scaler_model.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
        columns_to_scale = ['dpkts', 'sttl', 'smean', 'ct_srv_src', 'proto_target_encoded']
        X[columns_to_scale] = loaded_scaler.transform(X[columns_to_scale])
        return X



class Predictor(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        model = joblib.load('pretrained_voting2_model.pkl')
        pred =  model.predict(X)
        return pred