#for data manipulation
import pandas as pd
import sklearn

#for creating folder
import os

#for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

#for hugging face space authentication to upload files
from huggingface_hub import HfApi, login

#define constants for the dataset and output paths
api = HfApi(token= os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/rojasnath/Tourism-Project/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#Define target variable for the classification task
target = "ProdTaken"

#List of numerical features in the dataset
numeric_features = [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]

#Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

#Define target variable (y)
y = tourism_dataset[target]

#split dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train.to_csv("Xtrain.csv", index=False)
X_test.to_csv("Xtest.csv", index=False)
y_train.to_csv("ytrain.csv", index=False)
y_test.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path,
      repo_id="rojasnath/Tourism-Project",
      repo_type="dataset",
  )
