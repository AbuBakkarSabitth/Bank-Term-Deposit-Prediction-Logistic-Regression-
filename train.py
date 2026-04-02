import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('D:\\Problem-Set 2\\DataSet\\bank-full.csv', sep=';')
# Convert target
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encode categorical features
le = LabelEncoder()
for col in df.select_dtypes(include = ['object']).columns:
    df[col] = le.fit_transform(df[col])


# Split data
X = df.drop('y', axis = 1)
y = df['y']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
# Train model
model = LogisticRegression(max_iter = 1000)
model.fit(X_train,y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc  = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

import joblib
joblib.dump(model, "logistic_model.pkl")