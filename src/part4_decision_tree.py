'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def run_decision_tree():
    # Read in the dataframe(s) from PART 3
    df_arrests_train = pd.read_csv('../data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('../data/df_arrests_test.csv')

    # Create a list called `features` which contains our two feature names
    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    # Create a parameter grid called `param_grid_dt` containing three values for tree depth
    param_grid_dt = {'max_depth': [3, 5, 10]}

    # Initialize the Decision Tree model
    dt_model = DTC()

    # Initialize the GridSearchCV using the decision tree model and parameter grid
    gs_cv_dt = GridSearchCV(
        estimator=dt_model, 
        param_grid=param_grid_dt, 
        cv=KFold_strat(n_splits=5),  # 5-fold cross-validation with stratification
        scoring='accuracy'
    )

    # Run the model using the training set
    gs_cv_dt.fit(df_arrests_train[features], df_arrests_train['y'])

    # Get the best value for max_depth and print it
    best_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"The optimal value for max_depth: {best_max_depth}")

    # Predict for the test set
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(df_arrests_test[features])

    # Save the processed DataFrames for use in PART 5
    df_arrests_train.to_csv('../data/df_arrests_train_with_dt.csv', index=False)
    df_arrests_test.to_csv('../data/df_arrests_test_with_dt.csv', index=False)
