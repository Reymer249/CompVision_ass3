import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "jester-v1-train.csv"
data = pd.read_csv(file_path, sep=";")

# Split the data into train and validation sets with a 7:1 proportion
train_data, val_data = train_test_split(data, test_size=1/8, random_state=42)

# Save the split data into separate files
train_data.to_csv("train.csv", index=False, sep=";")
val_data.to_csv("val.csv", index=False, sep=";")

print("Files train.csv and val.csv created successfully!")