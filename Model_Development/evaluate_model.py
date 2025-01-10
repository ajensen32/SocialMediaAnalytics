import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from create_model_comparison_table import create_prediction_comparison_table

def evaluate_model(model):
    # Load the test data
    X_test = pd.read_csv("./X_test.csv")
    y_test = pd.read_csv("./y_test.csv").squeeze()
    X_train = pd.read_csv("./X_train.csv")
    y_train = pd.read_csv("./y_train.csv")

    # Load the saved encoder (if you need it for reference or saving purposes)
    encoder = joblib.load("./encoder.pkl")

    # Handle missing values (if any)
    X_test = X_test.fillna(X_test.mean())

    # Print out the columns to ensure consistency
    print("X_train columns:", X_train.columns)
    print("X_test columns:", X_test.columns)

    # Ensure that X_test has the same columns as X_train (in the same order)
    X_test = X_test[X_train.columns]  # Align columns to X_train

    # Apply the encoder to the test data
    categorical_columns = ['media_type_CAROUSEL_ALBUM', 'media_type_IMAGE', 'media_type_VIDEO', 
                           'time_of_day_0', 'time_of_day_1', 'time_of_day_2', 
                           'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday', 
                           'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 'day_of_week_Wednesday']
    
    # Ensure that the one-hot encoding process for categorical variables aligns with updated column names
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    # Create DataFrame for encoded features
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)

    # Drop original categorical columns and add encoded ones
    X_test = X_test.drop(columns=categorical_columns).join(X_test_encoded_df)

    # Handle missing values after encoding (if any)
    X_test = X_test.fillna(X_test.mean())

    # Make predictions
    y_pred = model.predict(X_test)

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("Actual Engagement Rate")
    plt.ylabel("Predicted Engagement Rate")
    plt.title("Actual vs Predicted Engagement Rate")
    plt.savefig("./actual_vs_predicted.png")  # Save plot to file
    plt.close()  # Close the plot to free resources

    # Analyze residuals (actual - predicted)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.savefig("./residuals_distribution.png")  # Save plot to file
    plt.close()  # Close the plot to free resources

    print("Evaluation completed. Visualizations saved.")
    create_prediction_comparison_table(y_test, y_pred)
