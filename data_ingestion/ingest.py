import sys
import os
import pandas as pd
import numpy as np
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)
import model_functions


# set seed for reproducibility
np.random.seed(123)
verbose = True

path = os.path.abspath("../data_ingestion/data")
save_path = os.path.abspath("../model_training/data")
# Load the data
def run_data_ingestion(path, save_path):
    data = model_functions.DataLoader(data_dir = path)    
    # model_functions.DataLoader.get_dataframe_info(data)

    data.model_df = data.modeling_df.drop_duplicates()

    data.select_features = data.model_df.drop(columns= ['label']).columns.tolist()

    data.monotonic_constraints = {}
    for col in data.select_features:
        data.monotonic_constraints[col] = 0

    data.model_df['label'] = (data.model_df['label'] - 1).astype(int)

    # data.model_df['scaling'] = (data.model_df['label'] + 1)/4

    # Split into training and prediction sets
    data.train_df = data.model_df.copy()
    data.prediction_set = data.model_df.copy()

    data.season_context = data.model_df[['season']] 

    # After processing your data, save all dataframes and check against an existing directory
    result = data.save_dataframes(save_dir=save_path, check_dir=path)

    # See what was saved and what was skipped
    if verbose:
        print(f"Saved dataframes: {result['saved']}")
        print(f"Skipped dataframes: {result['skipped']}")


# path = os.path.abspath("../rb_model/data_ingestion/data")
# save_path = os.path.abspath("../rb_model/model_training/data")
run_data_ingestion(path, save_path)