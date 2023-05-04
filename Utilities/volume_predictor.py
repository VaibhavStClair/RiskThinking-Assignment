# Importing necessary libraries
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Defining final data path
parent_dir = os.getcwd()
directory = "\\data\\Gold\\data.parquet"
final_data_path = parent_dir + directory
# Defining log paths
logs = "\\data\\logs"
final_log_path = parent_dir + logs


def volume_predictor(mov_avg, roll_med):
    """This function takes the features as moving average and rolling median of adj_close and target as Volume.
    The data is split and 20% is taken as test while 80% is taken as train.
    We apply Random Forrest Regression to predict the target.
    This function also stores the model results, log files and predicted values to the specific log folder
    and returns 3 values, predicted volume, moving average used and rolling median used.
    These values are later used for deployment in main file.
    """
    data_gold = pd.read_parquet(final_data_path)
    data_gold["Date"] = pd.to_datetime(data_gold["Date"])
    data_gold.set_index("Date", inplace=True)

    # Remove rows with NaN values
    data_gold.dropna(inplace=True)

    # Select features and target
    features = ["vol_moving_avg", "adj_close_rolling_med"]
    target = "Volume"

    X = data_gold[features]
    y = data_gold[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=10, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # predict test values

    test_values = model.predict([[mov_avg, roll_med]])

    # Calculate the Mean Absolute Error and Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # making log directory to store log files
    if not os.path.exists(final_log_path):
        os.makedirs(final_log_path)
    filename = final_log_path + "\\randomforestmodel.sav"
    pickle.dump(model, open(filename, "wb"))
    with open(final_log_path + "\\error_logs.txt", "w") as f:
        f.write(f"mean_absolute_error = {mae}, mean_squared_error {mse}")
    # adding predicted values to dataframe
    y_pred_df = pd.DataFrame(y_pred).reset_index(drop=True)
    y_pred_df.columns = ["Predicted"]
    # adding test values to dataframe
    y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
    y_test_df.columns = ["Actual"]
    # Concatenating both predicted and actual test values
    concat_df = pd.concat([y_test_df, y_pred_df], axis=1)
    concat_df.to_csv(final_log_path + "\\testdata_predicted_values.csv")

    return test_values, mov_avg, roll_med
