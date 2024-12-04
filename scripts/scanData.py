import pandas as pd

def print_csv_head(file_path, rows=5):
    try:
        data = pd.read_csv(file_path)
        print(data.head(rows))
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

csv_file_path = r"C:\Users\Ben Gur\src\src\wubai\data\fma_metadata\features.csv"
print_csv_head(csv_file_path, rows=5)
