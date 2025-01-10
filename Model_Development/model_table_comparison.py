import pandas as pd

def create_model_comparison_table(models_metrics, output_path="./model_comparison_table.csv"):
    """
    Create a comparison table for model metrics.

    Parameters:
        models_metrics (dict): A dictionary where keys are model names and values are their metrics.
                               Example:
                               {
                                   "Linear Regression": {"MAE": 0.02, "MSE": 0.0008, "R2": 0.85},
                                   "Random Forest": {"MAE": 0.01, "MSE": 0.0002, "R2": 0.92}
                               }
        output_path (str): Path to save the comparison table as a CSV file.

    Returns:
        None
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(models_metrics).T  # Transpose to have models as rows and metrics as columns

    # Save the table as a CSV file
    df.to_csv(output_path, index=True)
    print(f"Model comparison table saved to {output_path}")

    # Print the table for quick view
    print("\nModel Comparison Table:")
    print(df)

models_metrics = {
"Linear Regression": {"MAE": 0.0233, "MSE": 0.0008, "R2": -0.277},
"Random Forest": {"MAE": 0.0041, "MSE": 0.0000587, "R2": 0.9066},
"XGBoost": {"MAE": 0.0023, "MSE": 0.0000141, "R2": 0.9776},
"Gradient Boosting": {"MAE": 0.0038, "MSE": 0.0000310, "R2": 0.9507},
}

# Call the function to create and save the comparison table
create_model_comparison_table(models_metrics)