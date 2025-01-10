from train_advanced_models import train_advanced_models
from evaluate_model import evaluate_model
from model_table_comparison import create_model_comparison_table
import joblib


def main():
    print("Starting Model Development...\n")

    # Train advanced models and get metrics
    print("Training Advanced Models...\n")
    best_model, best_metrics, metrics_dict = train_advanced_models()

    # Create a comparison table
    print("\nCreating Model Comparison Table...")
    create_model_comparison_table(metrics_dict)

    # Evaluate the best model
    print("\nEvaluating Best Model...\n")
    evaluate_model(best_model)

    # Save the best model
    joblib.dump(best_model, "./best_model.pkl")
    print("Best model saved as 'best_model.pkl'")


if __name__ == "__main__":
    main()
