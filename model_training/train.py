import sys
import os
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, parent_dir)
import model_functions



# set seed for reproducibility
np.random.seed(123)
verbose = True


path = os.path.abspath("../model_training/data")
# Load the data
data = model_functions.DataLoader(path)    

# [data.X, data.y,
# data.train_X, data.train_y, data.train_sw, data.train_groups,
# data.test_X, data.test_y, data.test_sw, data.test_groups] = model_functions.prepare_model_splits_logo(data.train_df, select_features = data.select_features, stratify = True, target_col  = 'label')

data.train_X = data.model_df[data.model_df.season <= 2022][data.select_features].drop(columns= ['season'])
data.train_y = data.model_df[data.model_df.season <= 2022]['label']
data.train_groups = data.model_df[data.model_df.season <= 2022]['season']
# data.train_sw = data.model_df[data.model_df.season <= 2022]['scaling']


data.test_X = data.model_df[data.model_df.season >= 2023][data.select_features].drop(columns= ['season'])
data.test_y = data.model_df[data.model_df.season >= 2023]['label']
data.test_groups = data.model_df[data.model_df.season >= 2023]['season']
# data.test_sw = data.model_df[data.model_df.season >= 2023]['scaling']


data.plot_prediction_set = data.prediction_set[data.select_features]

data.full_predictions = data.model_df[data.select_features]

data.eval_model_df = data.model_df.copy()

data.best_model, data.feature_importances,data.y_proba, data.test_predictions, data.best_grid, data.results, data.best_group_scores = model_functions.xgboost_multiclass_model_logo(data.train_X, data.train_y, data.train_groups, data.test_X, data.test_y, 
                            # train_sw  = data.train_sw, test_sw  = data.test_sw,
                            monotonic_constraints=data.monotonic_constraints)



# Create a new MLflow Experiment
mlflow.set_experiment("playoff_prediction_model")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(data.best_grid)

    # Log the metrics
    for metric, value in data.results.items():
        mlflow.log_metric(metric, value)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Removing monotonic constraints")

    # Infer the model signature
    signature = infer_signature(data.train_X, data.best_model.predict(data.train_X))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=data.best_model,
        artifact_path="playoff_prediction_model",
        signature=signature,
        input_example=data.train_X,
        registered_model_name="playoff_prediction_model",
    )

    
    run = mlflow.active_run()
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    data.run_data  = [run_id, experiment_id]

# After processing your data, save all dataframes and check against an existing directory
result = data.save_dataframes(save_dir=os.path.abspath("../model_evaluation/data"))

# See what was saved and what was skipped
if verbose:
    print(f"Saved dataframes: {result['saved']}")
    print(f"Skipped dataframes: {result['skipped']}")