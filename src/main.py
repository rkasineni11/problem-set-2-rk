'''
You will run this problem set from main.py, so set things up accordingly
'''

import part1_etl
import part2_preprocessing
import part3_logistic_regression
import part4_decision_tree
import part5_calibration_plot

def main():
    # PART 1: Instantiate ETL, saving the two datasets in `./data/`
    part1_etl.run_etl()

    # PART 2: Call functions/instantiate objects from preprocessing
    part2_preprocessing.run_preprocessing()

    # PART 3: Call functions/instantiate objects from logistic_regression
    part3_logistic_regression.run_logistic_regression()

    # PART 4: Call functions/instantiate objects from decision_tree
    part4_decision_tree.run_decision_tree()

    # PART 5: Call functions/instantiate objects from calibration_plot
    part5_calibration_plot.run_calibration_plot()

if __name__ == "__main__":
    main()
