from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
from distances import haversine_array, dummy_manhattan_distance, bearing_array
from sklearn.cluster import MiniBatchKMeans
import pickle
import os
import matplotlib.pyplot as plt

class DataProcessing:
    def __init__(self):
        self.df = None

    def process_time(self):
        self.df['year'] = self.df['pickup_datetime'].dt.year
        self.df['month'] = self.df['pickup_datetime'].dt.month
        self.df['day'] = self.df['pickup_datetime'].dt.day
        self.df['hr'] = self.df['pickup_datetime'].dt.hour
        self.df['minute'] = self.df['pickup_datetime'].dt.minute

    # Weekends/Restdays Feature
    def restday(self, holidays):
        '''
        Output:
            is_rest: a list of Boolean variable indicating if the sample occurred in the rest day.
            is_weekend: a list of Boolean variable indicating if the sample occurred in the weekend.
        '''
        # Instanciate the rest and weekend lists of the size of the df dataframe
        is_holiday = [None] * len(self.df.year)
        is_weekend = [None] * len(self.df.year)

        '''
        i = 0
        for date_i in self.df.pickup_datetime:
            date_i = 
            # if current day is a weekend --> is_weekend[i] = True
            is_weekend[i] = date_i.isoweekday() in (6, 7)
            # if current day is a holiday --> is_rest[i] = True
            #  is_holiday[i] = datetime.timestamp(date_i) in holidays
            i += 1
        '''

        i = 0
        for yy, mm, dd in zip(self.df.year, self.df.month, self.df.day):
            # if current day is a weekend --> is_weekend[i] = True
            is_weekend[i] = date(yy, mm, dd).isoweekday() in (6, 7)
            # if current day is a holiday --> is_rest[i] = True
            is_holiday[i] = date(yy, mm, dd) in holidays
            i += 1
        return is_holiday, is_weekend

    def make_holidays(self):
        holidays = pd.read_csv('data/usholidays.csv', parse_dates=['Date'])
        holidays = holidays[holidays['Date'] > '2010-01-01']
        holidays['Date'] = holidays.Date.dt.date
        holidays = holidays.Date.to_list()

        # Find what dates are holidays and weekends
        is_holiday, is_weekend = self.restday(holidays)
        self.df['is_holiday'] = is_holiday
        self.df['is_weekend'] = is_weekend

    def make_distance(self):
        lat1, lng1, lat2, lng2 = (self.df['pickup_latitude'].values, self.df['pickup_longitude'].values,
                                  self.df['dropoff_latitude'].values, self.df['dropoff_longitude'].values)

        self.df['haversind_dist'] = haversine_array(lat1, lng1, lat2, lng2)
        self.df['manhattan_dist'] = dummy_manhattan_distance(lat1, lng1, lat2, lng2)
        self.df['bearing'] = bearing_array(lat1, lng1, lat2, lng2)

    def train_kmeans(self):
        # Cluster features
        # Stack together pickup and dropoff locations
        coords = np.vstack((self.df[['pickup_latitude', 'pickup_longitude']].values,
                            self.df[['dropoff_latitude', 'dropoff_longitude']].values))

        # Take 500,000 random indexes from the coords array
        sample_ind = np.random.permutation(len(coords))[:20000]

        # Fit a KMeans model with the sample data we just extracted and scaled
        kmeans = MiniBatchKMeans(n_clusters=10, batch_size=25).fit(coords[sample_ind])

        with open('model/kmeans.pkl', 'wb') as file:
            pickle.dump(kmeans, file)

        return kmeans

    def make_clusters(self, train=True):
        kmeans = None
        # if train = True, we retrain the KMeans model
        if train:
            kmeans = self.train_kmeans()
        # Otherwise, if the model has been trained once, we simply load the file
        # If it hasn't, we call the train_kmeans() function
        else:
            model_file = os.path.isfile('model/kmeans.pkl')
            if model_file:
                with open('model/kmeans.pkl', 'rb') as file:
                    kmeans = pickle.load(file)
            else:
                kmeans = self.train_kmeans()
                print('We train KMeans')

        # Make cluster prediction for pickup and dropoff postions (for train and test set)
        self.df.loc[:, 'pickup_cluster'] = kmeans.predict(self.df[['pickup_latitude', 'pickup_longitude']])
        self.df.loc[:, 'dropoff_cluster'] = kmeans.predict(self.df[['dropoff_latitude', 'dropoff_longitude']])

    def remove_outliers(self):
        # Removing outliers
        data_outliers = np.array([False] * len(self.df))

        # Number of passenger
        data_outliers[self.df.passenger_count > 6] = True

        # Taxi Fare
        data_outliers[self.df.fare_amount > 80] = True
        data_outliers[self.df.fare_amount < 0] = True

        # # Taxi locations
        data_outliers[self.df.pickup_longitude > -73] = True
        data_outliers[self.df.pickup_longitude < -74.5] = True

        data_outliers[self.df.pickup_latitude > 41.8] = True
        data_outliers[self.df.pickup_latitude < 40.5] = True


        data_outliers[self.df.dropoff_longitude > -73] = True
        data_outliers[self.df.dropoff_longitude < -74.5] = True

        data_outliers[self.df.dropoff_latitude > 41.8] = True
        data_outliers[self.df.dropoff_latitude < 40.5] = True

        mask = [not n for n in data_outliers]

        self.df = self.df.iloc[mask]


    def train_pipeline(self, df):
        self.df = df
        self.process_time()
        self.make_holidays()
        self.make_distance()
        self.train_kmeans()
        self.make_clusters(train=True)
        self.remove_outliers()
        return self.df