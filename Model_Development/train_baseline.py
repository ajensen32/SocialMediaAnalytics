import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

def train_baseline_model():
    # Load the data
    X_train = pd.read_csv("./X_train.csv")
    y_train = pd.read_csv("./y_train.csv").squeeze()
    X_test = pd.read_csv("./X_test.csv")
    y_test = pd.read_csv("./y_test.csv").squeeze()

    # Identify categorical columns (those that need one-hot encoding)
    categorical_columns = ['media_type_CAROUSEL_ALBUM', 'media_type_IMAGE', 'media_type_VIDEO', 
                           'time_of_day_0', 'time_of_day_1', 'time_of_day_2', 
                           'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday', 
                           'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday']


    # Perform one-hot encoding for categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Ensure that we use the same encoder to transform both training and testing data
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    # Create DataFrames for the encoded features
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

    # Drop original categorical columns and add encoded ones
    X_train = X_train.drop(columns=categorical_columns).join(X_train_encoded_df)
    X_test = X_test.drop(columns=categorical_columns).join(X_test_encoded_df)

    # Handle missing values: Fill NaN with column mean
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())  # Use training data's mean to avoid data leakage

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    return model, metrics
