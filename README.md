# housig-price-prediction
In this project we created a linear regression model that predicts the prices of houses in Boston area.
The dataset is housing_train.csv that have variables houseid,price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishing status.The new hose data is stored in housing_test.csv and the model produces predictions.csv having houseID and predicted prices for each house.Thw model predicts in 70% accuracy.
This project is part o the Edyst internship program.
code:
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm  

 

 

def binary_map(x):

    return x.map({'yes': 1, "no": 0})

 

def prepare_dataset(filename):

    housing = pd.read_csv(filename)

    varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    

    housing[varlist] = housing[varlist].apply(binary_map)

    

    # Let's drop the first column from status df using 'drop_first = True'

    status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)
Add the results to the original housing dataframe

    housing = pd.concat([housing, status], axis = 1)

    

    housing.drop(['furnishingstatus'], axis = 1, inplace = True)

    

    scaler = MinMaxScaler()

    num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

    housing[num_vars] = scaler.fit_transform(housing[num_vars])

 

    

    return housing

 def learn(filename):

    df_train = prepare_dataset(filename)

    

    y_train = df_train.pop('price')

    names = df_train.pop('houseID')

    X_train = df_train

    

    lm = LinearRegression()

    lm.fit(X_train, y_train)
rfe = RFE(lm, n_features_to_select=10)             # running RFE

    rfe = rfe.fit(X_train, y_train)

    

    col = X_train.columns[rfe.support_]

    X_train_rfe = X_train[col]

    

    

    X_train_rfe = sm.add_constant(X_train_rfe)

    

    lm = sm.OLS(y_train,X_train_rfe).fit()

    X_train_rfe = X_train_rfe.drop(['const'], axis=1)    

    return lm, X_train_rfe.columns

 

 

def predict(model, filename, columns):

    dataset = prepare_dataset(filename)

    X_test_new = dataset[columns]

    X_test_new = sm.add_constant(X_test_new)

    preds = model.predict(X_test_new)

    

    with_predictions = dataset.merge(preds.to_frame(), left_index=True, right_index=True)

    with_predictions = with_predictions.rename(columns={0:'predicted'})

    

    return with_predictions

 

model, columns= learn('housing_train.csv')

predictions = predict(model, 'housing_test.csv', columns)

 

predictions = predictions[['houseID', 'predicted']]

predictions.to_csv('predictions.csv', index=False)



 


 
