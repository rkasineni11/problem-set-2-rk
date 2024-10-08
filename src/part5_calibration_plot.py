'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, roc_auc_score

def run_calibration_plot():
    # Read in the dataframe(s) from PART 4
    df_arrests_test = pd.read_csv('../data/df_arrests_test_with_dt.csv')

    # Calibration plot function (already provided)
    def calibration_plot(y_true, y_prob, n_bins=10):
        bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
        sns.set(style="whitegrid")
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(prob_true, bin_means, marker='o', label="Model")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot")
        plt.legend(loc="best")
        plt.show()

    # 1. Calibration plot for Logistic Regression
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_lr'], n_bins=5)

    # 2. Calibration plot for Decision Tree
    calibration_plot(df_arrests_test['y'], df_arrests_test['pred_dt'], n_bins=5)

    # Extra Credit
    # 1. Compute PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
    top_50_lr = df_arrests_test.nlargest(50, 'pred_lr')
    ppv_lr = precision_score(top_50_lr['y'], top_50_lr['pred_lr'])

    # 2. Compute PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
    top_50_dt = df_arrests_test.nlargest(50, 'pred_dt')
    ppv_dt = precision_score(top_50_dt['y'], top_50_dt['pred_dt'])

    # 3. Compute AUC for the logistic regression model
    auc_lr = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_lr'])

    # 4. Compute AUC for the decision tree model
    auc_dt = roc_auc_score(df_arrests_test['y'], df_arrests_test['pred_dt'])

    # Print the results
    print(f"PPV for top 50 arrestees using Logistic Regression: {ppv_lr:.2f}")
    print(f"PPV for top 50 arrestees using Decision Tree: {ppv_dt:.2f}")
    print(f"AUC for Logistic Regression: {auc_lr:.2f}")
    print(f"AUC for Decision Tree: {auc_dt:.2f}")

    # 5. Do both metrics agree that one model is more accurate than the other?
    if (ppv_lr > ppv_dt) and (auc_lr > auc_dt):
        answer = "Yes, both metrics agree that the Logistic Regression model is more accurate."
    elif (ppv_lr < ppv_dt) and (auc_lr < auc_dt):
        answer = "Yes, both metrics agree that the Decision Tree model is more accurate."
    else:
        answer = "No, the metrics do not agree on which model is more accurate."

    print(f"Do both metrics agree that one model is more accurate than the other? {answer}")
