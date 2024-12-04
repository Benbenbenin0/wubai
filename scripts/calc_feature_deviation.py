import pandas as pd

features_df = pd.read_csv("data/generated_features.csv")

target_features = {
    "spectral_centroid_mean": 5000,
    "spectral_bandwidth_mean": 2000,
    "tempo": 128,
    "spectral_contrast_mean": 20,
}

for feature in target_features:
    target_value = target_features[feature]
    deviation_column = f"{feature}_deviation"
    features_df[deviation_column] = abs(features_df[feature] - target_value) / target_value

numeric_columns = [col for col in features_df.columns if features_df[col].dtype in ['float64', 'int64']]

grouped_deviations = features_df.groupby("epoch")[numeric_columns].mean()

grouped_deviations.to_csv("feature_deviations.csv")
print("Feature deviations calculated and saved to 'feature_deviations.csv'.")

print(grouped_deviations)
