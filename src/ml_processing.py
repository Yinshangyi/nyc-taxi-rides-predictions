from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd

scaler_link = 'generated/label_scaler.pickle'


class MLProcessing:
    def __init__(self):
        self.columns_cat = None
        self.columns_num = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def init_col_name(self):
        self.columns_cat = ['is_holiday', 'is_weekend', 'passenger_count', 'year', 'month',
                            'day', 'hr', 'minute', 'pickup_cluster', 'dropoff_cluster']

        self.columns_num = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                            'dropoff_latitude', 'haversind_dist', 'manhattan_dist',
                            'bearing', 'fare_amount']

        self.df = self.df[self.columns_cat + self.columns_num]

    def fillNaN(self):
        # Fill the NaNs of numerical data with the mean value of each column
        self.df[self.columns_num].fillna((self.df[self.columns_num].mean()), inplace=True)
        # Fill the NaNs of categorical data with the value of the previous data
        self.df[self.columns_num].fillna(method='ffill', inplace=True)

    def onehot(self):
        self.df = pd.get_dummies(self.df, columns=self.columns_cat)

    def scale(self):
        # Scale the features data
        scale_col = self.columns_num[:-1]
        scaler_features = MinMaxScaler()
        self.df[scale_col] = scaler_features.fit_transform(self.df[scale_col])

        # Scale the label
        scaler_label = MinMaxScaler()
        label_2D = self.df['fare_amount'].values.reshape(-1, 1)
        self.df['fare_amount'] = scaler_label.fit_transform(label_2D)

        file = open(scaler_link,'wb')
        pickle.dump(scaler_label, file)
        file.close()

    def split(self):
        X = self.df.drop('fare_amount', axis=1)
        y = self.df['fare_amount']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=55)

    @staticmethod
    def invert_scaling(values):
        values = values.reshape(-1,1)
        file = open(scaler_link, 'rb')
        scaler = pickle.load(file)
        file.close()

        descaled_values = scaler.inverse_transform(values)
        return descaled_values


    def transform(self, df):
        self.df = df
        self.init_col_name()
        self.fillNaN()
        self.onehot()
        self.scale()
        self.split()
        return self.X_train, self.X_test, self.y_train, self.y_test