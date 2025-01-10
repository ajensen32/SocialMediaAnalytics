import pandas as pd

def create_prediction_comparison_table(y_test, y_pred, output_path="./predictions_comparison.csv"):
    """
    Create a table comparing actual engagement values to predicted values.

    Parameters:
        y_test (pd.Series): The actual engagement values.
        y_pred (np.ndarray): The predicted engagement values.
        output_path (str): Path to save the comparison table as a CSV file.

    Returns:
        None
    """
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        "Actual Engagement": y_test,
        "Predicted Engagement": y_pred,
        "Residual": y_test - y_pred  # Difference between actual and predicted
    })

    # Save the table to a CSV file
    comparison_df.to_csv(output_path, index=False)
    print(f"Prediction comparison table saved to {output_path}")

    # Display the first few rows for quick review
    print("\nPrediction Comparison Table (Sample):")
    print(comparison_df.head())
