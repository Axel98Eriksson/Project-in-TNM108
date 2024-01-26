import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR  # Import Support Vector Regression
from sklearn.base import BaseEstimator, TransformerMixin

# Load HTML data
data = pd.read_html(r'data\train\sampdoria-s1.html', header=0, encoding='utf-8', keep_default_na=False)

# Assuming the first table in the HTML file is the one you want to use
# If there are multiple tables, you may need to inspect and select the correct one
data = data[0]

# Separate player names for visualization
player_names = data['Name']

# Drop player names and info from the features
#data = data.drop(['Inf','Name','Det','Fla','Agg'], axis=1)
data = data.drop(['Inf','Name'], axis=1)

#Change text to numeric values
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Position'])
data['Position'] = labelEncoder.transform(data['Position'])

#RIGHT/LEFT FOOT SYSTEM
data['Left Foot'] = data['Left Foot'].replace({'Very Strong' : '6','Strong' : '5', 'Fairly Strong' : '4', 'Reasonable' : '3', 'Weak' : '2', 'Very Weak' : '1'})
data['Right Foot'] = data['Right Foot'].replace({'Very Strong' : '6','Strong' : '5', 'Fairly Strong' : '4', 'Reasonable' : '3', 'Weak' : '2', 'Very Weak' : '1'})


# Convert the columns to numeric
data['Left Foot'] = pd.to_numeric(data['Left Foot'])
data['Right Foot'] = pd.to_numeric(data['Right Foot'])

data['Weakfoot Score'] = data['Left Foot'] + data['Right Foot']
data = data.drop(['Left Foot','Right Foot'], axis=1)


y = data['CA']
X = data.drop('CA', axis=1)

scaler = StandardScaler()
print(X)
data_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.75, random_state=42)


# Support Vector Machine Regression
svm_model = SVR(kernel='linear')  # You can experiment with different kernel functions
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate SVM Regression
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)
print(f'SVM Regression - Mean Squared Error: {mse_svm}')
print(f'SVM Regression - R-squared: {r2_svm}')

line = np.linspace(0, 200)

# Scatter plot of actual vs. predicted values for SVM Regression
plt.plot(line, line, color='red')
plt.scatter(y_test, y_pred_svm, alpha=0.2, color="white", edgecolors="orange", label='SVM Regression')
plt.title('Actual vs. Predicted Current Ability - SVM Regression')
plt.xlabel('Actual CA')
plt.ylabel('Predicted CA')
plt.legend()
plt.show()

# Save the SVM model for experimentation
#joblib.dump(svm_model, r'data\models\svm.joblib')
