import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

def train_advanced_models():
    # Load the data
    X_train = pd.read_csv("./X_train.csv")
    y_train = pd.read_csv("./y_train.csv").squeeze()
    X_test = pd.read_csv("./X_test.csv")
    y_test = pd.read_csv("./y_test.csv").squeeze()

    # List of columns already one-hot encoded and ready to use
    # Remove the old categorical columns that are now one-hot encoded
    categorical_columns = [
        'media_type_CAROUSEL_ALBUM', 'media_type_IMAGE', 'media_type_VIDEO',
        'time_of_day_0', 'time_of_day_1', 'time_of_day_2',
        'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
        'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday'
    ]

    # Boolean columns (already encoded as 1/0, so no further transformation needed)
    boolean_columns = ['contains_emoji', 'does_not_contain_emoji', 'contains_hashtag']

    # Convert boolean columns to integers (in case they are not already integers)
    for col in boolean_columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)

    # Fill missing values in numerical columns (if any)
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())

    # Make sure that only the relevant columns are used for training and testing
    X_train = X_train[categorical_columns + boolean_columns + ['likes', 'comments', 'shares', 'accounts_reached', 'followers_gained', 'hashtag_count', 'year', 'month', 'week', 'day', 'hour', 'caption_length']]
    X_test = X_test[categorical_columns + boolean_columns + ['likes', 'comments', 'shares', 'accounts_reached', 'followers_gained', 'hashtag_count', 'year', 'month', 'week', 'day', 'hour', 'caption_length']]

    # Initialize models
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    # Dictionary to store metrics
    metrics_dict = {}

    # Train and evaluate each model
    best_model = None
    best_metrics = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }
        metrics_dict[name] = metrics

        # Track the best model
        if best_metrics is None or metrics["R2"] > best_metrics["R2"]:
            best_model = model
            best_metrics = metrics

    # Return the best model and all metrics
    return best_model, best_metrics, metrics_dict


