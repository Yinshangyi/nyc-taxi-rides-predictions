from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from ml_processing import MLProcessing
import numpy as np
import pickle

class XGBoostModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, X_test, y_train, y_test):
        '''
        Trains the machine learning model based on the dataframe provided as input.
        The fitted model will be saved under model/xgboost.pkl
        The function returns the MSE and the RMSE
        :param df:
        :return: RMSE and MAE scores
        '''
        print('Training is starting...')
        eval_set = [(X_train, y_train), (X_test, y_test)]

        self.model = XGBRegressor(max_depth=7,
                                 objective='reg:squarederror',
                                 gamma=0,
                                 learning_rate=0.03,
                                 subsample=1,
                                 colsample_bytree=0.9,
                                 min_child_weight=10)

        self.model.fit(X_train, y_train,
                       eval_set=eval_set,
                       eval_metric="rmse",
                       early_stopping_rounds=500)

        predictions = self.predict(X_test)

        with open('generated/gxboost_model.pickle', 'wb') as file:
            pickle.dump(self.model, file)

        self.evaluate(y_test, X_test)


    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

    def grid_search(self, X_train, X_test, y_train, y_test):
        grid_param = {
            'max_depth': [n for n in range(2,10)],
            'gamma': np.arange(0, 0.5, 0.1),
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'subsample': np.arange(0.5, 0.9, 0.1),
            'colsample_bytree': np.arange(0.5, 0.9, 0.1),
            'min_child_weight': [1,3,5,7]
        }

        model = XGBRegressor(max_depth=7,
                                  objective='reg:squarederror',
                                  gamma=0,
                                  learning_rate=0.03,
                                  subsample=1,
                                  colsample_bytree=0.9,
                                  min_child_weight=10)

        gd_sr = GridSearchCV(estimator=model,
                             param_grid=grid_param,
                             scoring='neg_mean_squared_error',
                             cv=5,
                             n_jobs=-1)

        gd_sr.fit(X_train, y_train)

        best_parameters = gd_sr.best_params_
        print(best_parameters)

    def evaluate(self, y_test, X_test):
        print('#'*15+' Model Evaluation '+'#'*15)
        print()

        predictions = self.predict(X_test)
        predictions = MLProcessing.invert_scaling(predictions)
        y_test = MLProcessing.invert_scaling(np.array(y_test))

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        print('RMSE: {} - MAE: {}'.format(rmse, mae))

        print()
        print('#'*48)

