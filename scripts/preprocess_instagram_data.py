import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting

def add_engagement(input_file, output_file):
    data = pd.read_csv(input_file)
    data['engagement_rate'] = (data['likes'] + data['comments'] + data['shares']) / data['accounts_reached']
    data.to_csv(output_file)
    print(f"Processed data saved to {output_file}")








# Function to preprocess the data
def preprocess_instagram_data(input_file, output_file):
    data = pd.read_csv(input_file)
    data['contains_emoji'] = data['contains_emoji'].astype(int)
    data['does_not_contain_emoji'] = data['does_not_contain_emoji'].astype(int)
    data['contains_hashtag'] = data['contains_hashtag'].astype(int)
    
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['week'] = data['timestamp'].dt.isocalendar().week
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek + 1  # To get Monday=1, Sunday=7
    
    data['media_type_VIDEO'] = (data['media_type'] == 'VIDEO').astype(int)
    data['media_type_CAROUSEL'] = (data['media_type'] == 'CAROUSEL').astype(int)
    data['media_type_PHOTO'] = (data['media_type'] == 'PHOTO').astype(int)
    
    data['day_posted_monday'] = (data['day_of_week'] == 1).astype(int)
    data['day_posted_tuesday'] = (data['day_of_week'] == 2).astype(int)
    data['day_posted_wednesday'] = (data['day_of_week'] == 3).astype(int)
    data['day_posted_thursday'] = (data['day_of_week'] == 4).astype(int)
    data['day_posted_friday'] = (data['day_of_week'] == 5).astype(int)
    data['day_posted_saturday'] = (data['day_of_week'] == 6).astype(int)
    data['day_posted_sunday'] = (data['day_of_week'] == 7).astype(int)
    
    data['engagement_rate'] = (data['likes'] + data['comments'] + data['shares']) / data['accounts_reached']
    data['hashtag_count'] = data['caption'].apply(lambda x: str(x).count('#'))
    data['caption_length'] = data['caption'].apply(lambda x: len(str(x)))
    
    data.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    numerical_columns = ['likes', 'comments', 'shares', 'accounts_reached', 'followers_gained', 'engagement_rate', 'hashtag_count', 'caption_length']
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
    
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    data.set_index('timestamp', inplace=True)
    data = data.sort_index()
    
    final_columns = [
        'id', 'post_id', 'likes', 'comments', 'shares', 'contains_emoji', 'does_not_contain_emoji', 'contains_hashtag',
        'accounts_reached', 'followers_gained', 'engagement_rate', 'hashtag_count', 'year', 'month', 'week', 'day', 'hour', 
        'caption_length', 'day_of_week', 'media_type_VIDEO', 'media_type_CAROUSEL', 'media_type_PHOTO', 
        'day_posted_monday', 'day_posted_tuesday', 'day_posted_wednesday', 'day_posted_thursday', 
        'day_posted_friday', 'day_posted_saturday', 'day_posted_sunday'
    ]
    data = data[final_columns]
    
    data.to_csv(output_file)
    print(f"Processed data saved to {output_file}")

# Train-test split function
def train_test_split_by_timestamp(data, test_size=0.2, output_dir="./"):
    data = data.sort_index()
    cutoff_index = int(len(data) * (1 - test_size))
    
    train_data = data.iloc[:cutoff_index]
    test_data = data.iloc[cutoff_index:]
    
    X_train = train_data.drop(columns=['engagement_rate'])
    X_test = test_data.drop(columns=['engagement_rate'])
    y_train = train_data['engagement_rate']
    y_test = test_data['engagement_rate']

    # Ensure that X_train, y_train have consistent lengths
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    print(f"Train and test data saved to {output_dir}")
    return X_train, X_test, y_train, y_test

# Update functions to use .ravel() to reshape y_train and y_test
def train_baseline_model(X_train, X_test, y_train, y_test):
    y_train = y_train.values.ravel()  # Flatten y_train to 1D
    y_test = y_test.values.ravel()    # Flatten y_test to 1D
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Baseline Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    
    # Generate Graphical Reports
    generate_graphical_reports(model, X_train, y_train, X_test, y_test, model_name="Baseline_Model")

    return model, y_pred

def train_random_forest(X_train, X_test, y_train, y_test):
    # Ensure y_train and y_test are 1D arrays
    y_train = y_train.to_numpy().ravel()  # Flatten y_train to 1D
    y_test = y_test.to_numpy().ravel()    # Flatten y_test to 1D
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    
    # Evaluate the model
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae_rf}")
    print(f"Mean Squared Error (MSE): {mse_rf}")
    print(f"R-squared (R²): {r2_rf}")
    
    # Generate Graphical Reports
    generate_graphical_reports(rf_model, X_train, y_train, X_test, y_test, model_name="Random_Forest")

    # Ensure that the model and predictions are returned
    return rf_model, y_pred_rf



def train_xgboost(X_train, X_test, y_train, y_test):
    # Initialize the XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred_xgb = xgb_model.predict(X_test)

    # Evaluate the model
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    print(f"XGBoost Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae_xgb}")
    print(f"Mean Squared Error (MSE): {mse_xgb}")
    print(f"R-squared (R²): {r2_xgb}")
    
    # Generate Graphical Reports
    generate_graphical_reports(xgb_model, X_train, y_train, X_test, y_test, model_name="XGBoost_Model")

    return xgb_model, y_pred_xgb



# def train_lightgbm(X_train, X_test, y_train, y_test):
#     # Initialize the LightGBM model
#     lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

#     # Train the model
#     lgb_model.fit(X_train, y_train)

#     # Make predictions
#     y_pred_lgb = lgb_model.predict(X_test)

#     # Evaluate the model
#     mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
#     mse_lgb = mean_squared_error(y_test, y_pred_lgb)
#     r2_lgb = r2_score(y_test, y_pred_lgb)

#     # Print evaluation metrics
#     print(f"LightGBM Model Evaluation:")
#     print(f"Mean Absolute Error (MAE): {mae_lgb}")
#     print(f"Mean Squared Error (MSE): {mse_lgb}")
#     print(f"R-squared (R²): {r2_lgb}")

#     return lgb_model, y_pred_lgb


def tune_random_forest(X_train, X_test, y_train, y_test):
    # Define the model
    rf_model = RandomForestRegressor(random_state=42)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the model to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Print the best parameters and score
    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validation score: {best_score}")

    # Get the best model from grid search
    best_rf_model = grid_search.best_estimator_

    # Make predictions using the best model
    y_pred_rf = best_rf_model.predict(X_test)

    # Evaluate the model
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest Model Evaluation after Hyperparameter Tuning:")
    print(f"Mean Absolute Error (MAE): {mae_rf}")
    print(f"Mean Squared Error (MSE): {mse_rf}")
    print(f"R-squared (R²): {r2_rf}")

    return best_rf_model, y_pred_rf


def cross_validate_random_forest(X_train, y_train, X_test, y_test):
    # Define the model
    rf_model = RandomForestRegressor(random_state=42)
    
    # Define the parameter grid to search over
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define GridSearchCV with cross-validation (e.g., 5-fold)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

    # Fit the model to the training data with cross-validation
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best model from GridSearchCV
    best_params = grid_search.best_params_
    best_rf_model = grid_search.best_estimator_

    # Print the best parameters
    print(f"Best Parameters: {best_params}")
    
    # Make predictions using the best model
    y_pred = best_rf_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation with Cross-Validation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    
    return best_rf_model, y_pred





# Function to perform cross-validation with RandomizedSearchCV for RandomForest
def randomized_search_random_forest(X_train, y_train, X_test, y_test):
    # Ensure y_train and y_test are 1D arrays
    y_train = y_train.values.ravel()  # Convert DataFrame to 1D array and flatten
    y_test = y_test.values.ravel()    # Convert DataFrame to 1D array and flatten
    
    # Define the model
    rf_model = RandomForestRegressor(random_state=42)
    
    # Define the parameter grid with random distribution
    param_dist = {
        'n_estimators': np.arange(100, 500, 100),
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }

    # Define RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                       n_iter=100, cv=5, scoring='neg_mean_squared_error', 
                                       n_jobs=-1, verbose=2, random_state=42)

    # Fit the model to the training data with cross-validation
    random_search.fit(X_train, y_train)

    # Get the best parameters and best model from RandomizedSearchCV
    best_params = random_search.best_params_
    best_rf_model = random_search.best_estimator_

    # Print the best parameters
    print(f"Best Parameters: {best_params}")
    
    # Make predictions using the best model
    y_pred = best_rf_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation with Randomized Search:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")
    
    return best_rf_model, y_pred


def generate_graphical_reports(model, X_train, y_train, X_test, y_test, model_name="Model", output_dir="./"):
    # Feature Importance Visualization (Random Forest, XGBoost, etc.)
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.barh(X_train.columns, feature_importance)
        plt.title(f'{model_name} - Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_feature_importance.png")  # Save plot as PNG
        plt.close()  # Close the figure to prevent display

    # Predictions vs Actual plot
    y_pred = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.title(f'{model_name} - Predictions vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_predictions_vs_actual.png")  # Save plot as PNG
    plt.close()  # Close the figure to prevent display

    # Model Evaluation Metrics (Bar chart for MAE, MSE, R²)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'MAE': mae, 'MSE': mse, 'R²': r2}
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green'])
    plt.title(f'{model_name} - Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_evaluation_metrics.png")  # Save plot as PNG
    plt.close()  # Close the figure to prevent display

    print(f"Graphs saved to {output_dir}")

def evaluate_model(y_test, y_pred, model_name="Random Forest"):
    """
    Evaluates the model using MAE, MSE, and R-squared metrics.
    Also generates a plot comparing the actual vs predicted values.
    
    Parameters:
        y_test (array-like): Actual values of the target variable (engagement rate).
        y_pred (array-like): Predicted values of the target variable.
        model_name (str): Name of the model being evaluated (for display purposes).
    
    Returns:
        None
    """
    # Calculate the evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the evaluation metrics
    print(f"{model_name} Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R²): {r2}")

    # Visualize the predictions vs actual values
    plt.figure(figsize=(10, 6))

    # Scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')

    # Plot the identity line (x = y), indicating perfect predictions
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label="Perfect Prediction Line")
    
    plt.title(f"{model_name} - Actual vs Predicted Engagement")
    plt.xlabel('Actual Engagement')
    plt.ylabel('Predicted Engagement')
    plt.legend()
    
    # Save the plot to a PNG file
    plt.savefig(f"{model_name}_actual_vs_predicted.png")
    print(f"Graph saved to {model_name}_actual_vs_predicted.png")
    
    # Show the plot
    plt.show()



def load_data(input_file):

    data = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)
    return data








# X_train = pd.read_csv("./X_train.csv")
# y_train = pd.read_csv("./y_train.csv")
# X_test = pd.read_csv("./X_test.csv")
# y_test = pd.read_csv("./y_test.csv")

# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

# # Example usage:
# # Assuming you have already trained a model and have predictions
# rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test)
# # Nevaluate the model:
# evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

















# # Example usage
# input_file = "./Instagram_data3.csv"
# output_file = "./combined_data.csv"
# preprocess_instagram_data(input_file, output_file)




# # Example usage:
# input_file = "./combined_data.csv"
# data = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)
# # Specify the directory where you want to save the CSV files
# output_dir = "./"  # Save in the current directory
# X_train, X_test, y_train, y_test = train_test_split_by_timestamp(data, output_dir=output_dir)




# # Example usage:
# # Load the training and testing data
# X_train = pd.read_csv("./X_train.csv").values.ravel()
# X_test = pd.read_csv("./X_test.csv").values.ravel()
# y_train = pd.read_csv("./y_train.csv").values.ravel()
# y_test = pd.read_csv("./y_test.csv").values.ravel()
# # Train the baseline model
# model, y_pred = train_baseline_model(X_train, X_test, y_train, y_test)



# # Example usage:
# rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test)


# Example usage:
# # xgb_model, y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test)



# # # Example usage:
# # # lgb_model, y_pred_lgb = train_lightgbm(X_train, X_test, y_train, y_test)

# # # Example usage:
# # best_rf_model, y_pred_rf = tune_random_forest(X_train, X_test, y_train, y_test)


# # # Example usage:
# # # # Assume X_train, X_test, y_train, and y_test are already prepared
# # rf_model, y_pred_rf = cross_validate_random_forest(X_train, y_train, X_test, y_test)



# # # Example usage:
# # # Assume X_train, X_test, y_train, and y_test are already prepared
# # rf_model, y_pred_rf = randomized_search_random_forest(X_train, y_train, X_test, y_test)



# # # Example usage:
# # input_file = "./combined_data.csv"
# # data = pd.read_csv(input_file, index_col="timestamp", parse_dates=True)
# # output_dir = "./"  # Specify where to save the images
# # X_train, X_test, y_train, y_test = train_test_split_by_timestamp(data, output_dir=output_dir)

# # Train the Baseline Model and generate reports
# model, y_pred = train_baseline_model(X_train, X_test, y_train, y_test)

# Train the Random Forest Model and generate reports
# rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test)

# # Train the XGBoost Model and generate reports
# xgb_model, y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test)






# Test the functions

# Load the data
# data = load_data("./combined_data.csv")

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split_by_timestamp(data, output_dir="./")

# # Train and evaluate the baseline model (Linear Regression)
# model, y_pred = train_baseline_model(X_train, X_test, y_train, y_test)
# evaluate_model(y_test, y_pred, model_name="Baseline Model")

# # Train and evaluate the Random Forest model
# rf_model, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test)
# evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

# # Train and evaluate the XGBoost model
# xgb_model, y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test)
# evaluate_model(y_test, y_pred_xgb, model_name="XGBoost")

# # Generate graphical reports
# generate_graphical_reports(rf_model, X_train, y_train, X_test, y_test, model_name="Random Forest", output_dir="./")
# generate_graphical_reports(xgb_model, X_train, y_train, X_test, y_test, model_name="XGBoost", output_dir="./")



# def plot_likes_vs_engagement(data):
#     """
#     Plots the relationship between Likes and Engagement Rate.
#     """
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=data['likes'], y=data['engagement_rate'])
#     plt.title('Likes vs Engagement Rate')
#     plt.xlabel('Likes')
#     plt.ylabel('Engagement Rate')
#     plt.tight_layout()
#     plt.savefig("./likes_vs_engagement.png")
#     plt.close()

# # Usage example
# plot_likes_vs_engagement(data)

def plot_model_comparison(models, r2_scores):
    """
    Plots the comparison of R² scores for different models.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=r2_scores, palette='Blues_d')
    plt.title('Model Comparison - R² Scores')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.tight_layout()
    plt.savefig("./model_comparison.png")
    plt.close()

# Usage example
models = ['Linear Regression', 'Random Forest', 'XGBoost']
r2_scores = [-9.26, 0.81, 0.88]
plot_model_comparison(models, r2_scores)





