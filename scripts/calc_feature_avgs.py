import pandas as pd
import numpy as np

def clean_and_convert(column):
    cleaned_values = []
    for entry in column:
        if isinstance(entry, str) and entry.startswith("[") and entry.endswith("]"):
            try:
                parsed_list = np.array(eval(entry))
                cleaned_values.append(parsed_list.mean())
            except Exception:
                cleaned_values.append(np.nan)
        else:
            try:
                cleaned_values.append(float(entry))
            except ValueError:
                cleaned_values.append(np.nan)
    return pd.Series(cleaned_values)

def calculate_averages(features_csv):
    features_df = pd.read_csv(features_csv)

    columns_to_average = [
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "tempo",
        "spectral_contrast_mean"
    ]

    missing_columns = [col for col in columns_to_average if col not in features_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV file: {missing_columns}")

    averages = {}
    for column in columns_to_average:
        features_df[column] = clean_and_convert(features_df[column])
        averages[column] = features_df[column].mean() 

    return averages

if __name__ == "__main__":
    features_csv_path = "data/processed_techno_features.csv"

    try:
        averages = calculate_averages(features_csv_path)

        print("\nAveraged Feature Values:")
        for feature, avg_value in averages.items():
            print(f"  {feature}: {avg_value:.4f}")
    except Exception as e:
        print(f"Error: {e}")
