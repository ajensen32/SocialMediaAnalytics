import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib



def add_engagement_metrics(csv_path: str, output_path: str):
    """
    Adds engagement rate, caption length, and hashtag count to the CSV file.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path to save the updated CSV file.
    """
    # Load the data
    data = pd.read_csv(csv_path)
    
    # Calculate engagement rate
    # Ensure no division by zero by replacing 0 accounts_reached with NaN
    data['accounts_reached'] = data['accounts_reached'].replace(0, pd.NA)
    data['engagement_rate'] = (
        (data['likes'] + data['comments'] + data['shares']) / data['accounts_reached']
    ).fillna(0)  # Replace NaN with 0 for engagement rate
    
    # Calculate caption length
    # data['caption_length'] = data['caption'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    
    # Count hashtags in captions
    # data['hashtag_count'] = data['caption'].apply(lambda x: str(x).count('#') if pd.notna(x) else 0)
    
    # Save the updated data back to a new CSV file
    data.to_csv(output_path, index=False)
    print(f"Updated CSV saved to {output_path}")



def process_csv_times_series(file_path, output_path):
    """
    Process the CSV file by reordering columns and adding new time-related columns.
    
    Args:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to save the processed CSV file.
    """
    try:
        # Load the CSV file
        csv_path = Path(file_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        data = pd.read_csv(csv_path)

        # Check if 'timestamp' exists in the CSV
        if 'timestamp' not in data.columns:
            raise KeyError("The 'timestamp' column is missing in the CSV file.")

        # Convert 'timestamp' to datetime for further processing
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

        # Add new columns based on timestamp
        data['year'] = data['timestamp'].dt.year
        data['month'] = data['timestamp'].dt.month
        data['week'] = data['timestamp'].dt.isocalendar().week
        data['day'] = data['timestamp'].dt.day
        data['hour'] = data['timestamp'].dt.hour
        # data['day_of_week'] = data['timestamp'].dt.day_name()

        # Reorder columns to place 'timestamp' as the first column
        columns = ['timestamp'] + [col for col in data.columns if col != 'timestamp']
        data = data[columns]

        # Save the updated DataFrame to the output path
        data.to_csv(output_path, index=False)
        print(f"CSV file has been updated and saved to {output_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")




def normalize_data(filepath, output_path):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Print out column names to debug
    print("Columns in dataset:", data.columns)
    
    # Separate columns into numerical, categorical, and boolean
    numeric_columns = ['likes', 'comments', 'shares', 'accounts_reached', 'followers_gained', 'hashtag_count', 'caption_length', 'year', 'month', 'week', 'day', 'hour']
    
    # Ensure the columns exist before adding them to the list
    categorical_columns = [
        'media_type_CAROUSEL_ALBUM', 'media_type_IMAGE', 'media_type_VIDEO', 
        'time_of_day_0', 'time_of_day_1', 'time_of_day_2',
        'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday', 
        'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday', 
        'day_of_week_Wednesday'
    ]
    
    # Check if the columns are in the dataset before proceeding
    missing_categorical_columns = [col for col in categorical_columns if col not in data.columns]
    if missing_categorical_columns:
        print(f"Warning: Missing categorical columns: {missing_categorical_columns}")
    
    # Filter out missing columns from the list before applying one-hot encoding
    categorical_columns = [col for col in categorical_columns if col in data.columns]

    # Boolean columns
    boolean_columns = ['contains_emoji', 'does_not_contain_emoji', 'contains_hashtag']
    
    # Handle boolean columns by converting them to integers (True=1, False=0)
    for col in boolean_columns:
        if col in data.columns:
            data[col] = data[col].astype(int)

    # Fill NaN values for numeric columns with their respective mean
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mean())

    # One-hot encode categorical columns if they exist
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if categorical_columns:  # Only proceed if categorical columns are present
        encoded_features = encoder.fit_transform(data[categorical_columns])
        # Create a DataFrame for the encoded features
        encoded_columns = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=data.index)
        
        # Drop original categorical columns and add the encoded ones
        data = data.drop(columns=categorical_columns).join(encoded_df, rsuffix='_encoded')

    # Clean up column names: remove the "_0.0" and "_1.0" suffixes
    data.columns = [col.split('_')[0] if '_0.0' in col or '_1.0' in col else col for col in data.columns]

    # Normalize numerical columns to be between 0 and 1
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # Save the normalized data to a new CSV file
    data.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")

    return data


def split_data(data_file_path):
    # Load the data
    data = pd.read_csv(data_file_path)

    # Ensure timestamp is in datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Calculate 'caption_length' as the number of characters in each caption
    # data['caption_length'] = data['caption'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

    # Drop unnecessary columns (keep 'caption_length')
    # data = data.drop(['caption'], axis=1, errors='ignore')  # 'caption' column can be dropped after 'caption_length' is created

    # Sort the data by timestamp to maintain chronological order
    data = data.sort_values(by='timestamp').reset_index(drop=True)

    # Drop unnecessary columns (id, post_id)
    X = data.drop(['engagement_rate', 'timestamp', 'id', 'post_id'], axis=1)
    y = data['engagement_rate']

    # Calculate the split index for an 80-20 split
    split_index = int(len(data) * 0.8)

    # Perform the split
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Save the split datasets to CSV files
    X_train.to_csv("./X_train.csv", index=False)
    X_test.to_csv("./X_test.csv", index=False)
    y_train.to_csv("./y_train.csv", index=False)
    y_test.to_csv("./y_test.csv", index=False)

    # Print the shapes of the resulting datasets to verify
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")
    print(f"Training Target Shape: {y_train.shape}")
    print(f"Testing Target Shape: {y_test.shape}")
    print("Data split and saved successfully!")

#feature selection

matplotlib.use('Agg')  # Use a non-interactive backend

def load_data(filepath="./combined_data.csv"):
    # Load data from CSV
    data = pd.read_csv(filepath)
    return data

#Correlation Analysis

def correlation_analysis(data, target='engagement_rate', threshold=0.5, output_path="./correlation_heatmap.png"):
    """
    Perform correlation analysis and save the heatmap to a file.

    Parameters:
        data (pd.DataFrame): The input data.
        target (str): The target variable for correlation.
        threshold (float): The correlation threshold for feature selection.
        output_path (str): The path to save the correlation heatmap image.

    Returns:
        pd.DataFrame: Data filtered to only include relevant features.
    """
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=[np.number])

    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()

    # Save heatmap as a PNG file
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)  # Save the heatmap as an image
    plt.close()  # Close the plot to free up resources

    print(f"Correlation heatmap saved to {output_path}")

    # Identify features with high correlation with the target variable
    corr_with_target = correlation_matrix[target].sort_values(ascending=False)
    relevant_features = corr_with_target[corr_with_target.abs() > threshold].index.tolist()

    print(f"Features with correlation > {threshold}:\n{relevant_features}")

    # Return the filtered DataFrame
    return data[relevant_features + [target]]



#Mutal Information Analysis
def mutual_information_analysis(data, target='engagement_rate'):
    """
    Perform mutual information analysis to identify feature importance.

    Parameters:
        data (pd.DataFrame): The input data.
        target (str): The target variable for feature selection.

    Returns:
        pd.Series: Mutual information scores for features.
    """
    # Ensure target is a single-dimensional array
    X = data.loc[:, data.columns != target]  # Drop target column from features
    y = data[target]  # Target as 1D array

    # Check if X and y have any missing or incorrect data
    if X.isnull().sum().sum() > 0:
        print("Warning: Missing values in feature set. Handling missing values.")
        X = X.fillna(X.mean())  # Fill missing values with mean
    
    if y.isnull().sum() > 0:
        print("Warning: Missing values in target variable. Handling missing values.")
        y = y.fillna(y.mean())  # Fill missing values in target variable

    # Ensure X and y are properly shaped (1D for target, 2D for features)
    if X.empty or y.empty:
        print("Error: Feature set (X) or target variable (y) is empty.")
        return None

    # Compute mutual information scores
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Plot the scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='bar')
    plt.title("Mutual Information Scores")
    plt.savefig("./mutual_information_scores.png")  # Save plot as image
    plt.close()  # Close the plot

    print("Mutual Information Scores:\n", mi_scores)
    print("Mutual information scores saved to './mutual_information_scores.png'")

    return mi_scores

#Permutation Importance

def permutation_importance_analysis(data, target='engagement_rate'):
    X = data.drop(columns=[target])
    y = data[target]

    # Check if X and y are empty or have missing values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    if X.isnull().sum().sum() > 0:
        print("Warning: Missing values in feature set. Filling missing values with mean.")
        X = X.fillna(X.mean())  # Fill missing values in X with column means

    if y.isnull().sum() > 0:
        print("Warning: Missing values in target variable. Filling missing values with mean.")
        y = y.fillna(y.mean())  # Fill missing values in y with mean

    # Ensure X and y are properly formatted
    if X.empty or y.empty:
        print("Error: Feature set (X) or target variable (y) is empty.")
        return None

    # Check if y is a 1D array (series) and X is a 2D DataFrame
    if not isinstance(X, pd.DataFrame):
        print("Error: X is not a DataFrame.")
        return None
    if not isinstance(y, pd.Series):
        print("Error: y is not a Series.")
        return None
    
    # Initialize the model (Random Forest)
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # Compute permutation importance
    perm_importance = permutation_importance(model, X, y, random_state=42)
    importance_df = pd.DataFrame(perm_importance.importances_mean, index=X.columns, columns=["Importance"]).sort_values(by="Importance", ascending=False)

    # Plot the importance
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    importance_df.plot(kind='bar')
    plt.title("Permutation Feature Importance")
    plt.show()

    print("Permutation Importance Scores:\n", importance_df)
    return importance_df


#Integrate Feature Selection into the Pipline
def select_features(filepath="./combined_data.csv", target='engagement_rate'):
    # Load data
    data = load_data(filepath)

    # Correlation Analysis
    print("Running Correlation Analysis...")
    corr_data = correlation_analysis(data, target=target, threshold=0.3)

    # Ensure no duplicate columns in `corr_data`
    corr_data = corr_data.loc[:, ~corr_data.columns.duplicated()]

    # Mutual Information Analysis
    print("Running Mutual Information Analysis...")
    mi_scores = mutual_information_analysis(corr_data, target=target)

    # Permutation Importance
    print("Running Permutation Importance Analysis...")
    perm_scores = permutation_importance_analysis(corr_data, target=target)

    # Select final features based on permutation and mutual information
    final_features = perm_scores.index[perm_scores['Importance'] > 0.01].tolist()
    print(f"Selected Features: {final_features}")

    # Save filtered dataset
    selected_data = corr_data[final_features + [target]]
    selected_data.to_csv("selected_features_data.csv", index=False)
    print("Filtered dataset saved as 'selected_features_data.csv'")


#Run Feature Selection









filepath="./combined_data.csv"

output_path="./combined_data.csv"

add_engagement_metrics(filepath, output_path)
process_csv_times_series(output_path, output_path)
normalize_data(output_path, output_path)
split_data(output_path)
select_features(filepath="./combined_data.csv", target="engagement_rate")









