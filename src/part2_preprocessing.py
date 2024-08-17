'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

import pandas as pd

def run_preprocessing():
    # Load the data from the saved CSV files
    pred_universe = pd.read_csv('../data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('../data/arrest_events_raw.csv')

    # Ensure 'arrest_date_univ' and 'arrest_date_event' are datetime objects
    pred_universe['arrest_date_univ'] = pd.to_datetime(pred_universe['arrest_date_univ'])
    arrest_events['arrest_date_event'] = pd.to_datetime(arrest_events['arrest_date_event'])

    # Perform a full outer join on 'person_id'
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # Print the columns to identify the correct one
    print("Columns in df_arrests DataFrame:")
    print(df_arrests.columns)

    # Create the 'current_charge_felony' column based on 'charge_degree'
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x.lower() == 'felony' else 0)

    # Create the 'y' column for felony arrests within 365 days after the current arrest
    df_arrests['y'] = df_arrests.apply(lambda row: int(
        pd.notna(row['arrest_date_event']) and
        (row['arrest_date_event'] > row['arrest_date_univ']) and 
        (row['arrest_date_event'] <= (row['arrest_date_univ'] + pd.Timedelta(days=365))) and 
        (row['current_charge_felony'] == 1)
    ), axis=1)

    # Question 1: What share of arrestees were rearrested for a felony crime in the next year?
    share_rearrested = df_arrests['y'].mean()
    print(f"What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year? {share_rearrested:.2%}")

    # Question 2: What share of current charges are felonies?
    share_felonies = df_arrests['current_charge_felony'].mean()
    print(f"What share of current charges are felonies? {share_felonies:.2%}")

    # Create the predictive feature 'num_fel_arrests_last_year'
    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(lambda row: df_arrests[
        (df_arrests['person_id'] == row['person_id']) &
        (df_arrests['arrest_date_event'] < row['arrest_date_event']) &
        (df_arrests['arrest_date_event'] >= (row['arrest_date_event'] - pd.Timedelta(days=365))) &
        (df_arrests['current_charge_felony'] == 1)
    ].shape[0], axis=1)

    # Question 3: What is the average number of felony arrests in the last year?
    average_felony_arrests = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_felony_arrests:.2f}")

    # Print the mean of 'num_fel_arrests_last_year'
    print(f"Mean of 'num_fel_arrests_last_year': {df_arrests['num_fel_arrests_last_year'].mean()}")

    # Print the head of df_arrests
    print(df_arrests.head())

    # Save the processed DataFrame for use in PART 3 or return it
    df_arrests.to_csv('../data/df_arrests.csv', index=False)
