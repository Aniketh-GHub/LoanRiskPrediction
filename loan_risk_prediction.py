import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = r'C:\Users\aniketh\Downloads\drive-download-20240615T064342Z-001\data\loan_approval_dataset.json'
data = pd.read_json(file_path)
# Check for missing values
data.isnull().sum()
# Handle missing values for numeric columns
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
# One-hot encode categorical columns
data_encoded = pd.get_dummies(data)
# Split the data into training and testing sets
X = data_encoded.drop(columns=['Risk_Flag'])
y = data_encoded['Risk_Flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(report)
# Plot a histogram 
data = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
plt.hist(data)
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(data['Loan_Amount'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Loan Amounts')
plt.show()
