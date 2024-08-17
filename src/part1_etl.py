'''
PART 1: ETL the two datasets and save each in `data/` as .csv's
'''

import pandas as pd

def run_etl():
    # Load the datasets from the URLs
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

    # Convert the filing dates to datetime and remove the original 'filing_date' column
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])

    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    # Save the DataFrames as CSV files in the 'data/' directory
    pred_universe_raw.to_csv('../data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('../data/arrest_events_raw.csv', index=False)

