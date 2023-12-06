import joblib
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# Load HTML data
data = pd.read_html(r'data\train\train-model-v0.html', header=0, encoding='utf-8', keep_default_na=False)

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

labelEncoder.fit(data['Right Foot'])
data['Right Foot'] = labelEncoder.transform(data['Right Foot'])

labelEncoder.fit(data['Left Foot'])
data['Left Foot'] = labelEncoder.transform(data['Left Foot'])

y = data['CA']
X = data.drop('CA', axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.75, random_state=42)

# Create and fit the linear regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#accuracy = accuracy_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
#print(f'Accuracy: {accuracy}')

# Scatter plot of actual vs. predicted values
line = np.linspace(0,200)
plt.plot(line,line, color = 'red')
plt.scatter(y_test, y_pred, alpha=0.2, color="white", edgecolors="blue")
plt.title('Actual vs. Predicted Current Ability')
plt.xlabel('Actual CA')
plt.ylabel('Predicted CA')
plt.show()

#save the model for experimentation
joblib.dump(model, r'data\models\v02.joblib')