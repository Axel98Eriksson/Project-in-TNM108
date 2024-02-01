import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# Load HTML data
data = pd.read_html(r'data\train\attackers.html', header=0, encoding='utf-8', keep_default_na=False)


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

# Linear Regression
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Ridge Regression
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)



# Evaluate Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test,y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R-squared: {r2_linear}')
print(f'Linear Regression - Mean Absolute Error: {mae_linear}')

# Evaluate Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test,y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f'Ridge Regression - Mean Squared Error: {mse_linear}')
print(f'Ridge Regression - Mean Absolute Error: {mae_ridge}')
print(f'Ridge Regression - R-squared: {r2_ridge}')

# Evaluate Lasso Regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test,y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'Lasso Regression - Mean Squared Error: {mse_linear}')
print(f'Lasso Regression - Mean Absolute Error: {mae_lasso}')
print(f'Lasso Regression - R-squared: {r2_lasso}')

# Scatter plot of actual vs. predicted values for Linear Regression
line = np.linspace(0, 200)
plt.plot(line, line, color='red')
plt.scatter(y_test, y_pred_linear, alpha=0.2, color="white", edgecolors="blue", label='Linear Regression')
plt.title('Actual vs. Predicted Current Ability - Linear Regression')
plt.xlabel('Actual CA')
plt.ylabel('Predicted CA')
plt.legend()
plt.show()

# Scatter plot of actual vs. predicted values for Ridge Regression
plt.plot(line, line, color='red')
plt.scatter(y_test, y_pred_ridge, alpha=0.2, color="white", edgecolors="green", label='Ridge Regression')
plt.title('Actual vs. Predicted Current Ability - Ridge Regression')
plt.xlabel('Actual CA')
plt.ylabel('Predicted CA')
plt.legend()
plt.show()

# Scatter plot of actual vs. predicted values for Lasso Regression
plt.plot(line, line, color='red')
plt.scatter(y_test, y_pred_lasso, alpha=0.2, color="white", edgecolors="purple", label='Lasso Regression')
plt.title('Actual vs. Predicted Current Ability - Lasso Regression')
plt.xlabel('Actual CA')
plt.ylabel('Predicted CA')
plt.legend()
plt.show()

# Save the models for experimentation
#joblib.dump(linear_model, r'data\models\sampdoria-s1.joblib')
#joblib.dump(ridge_model, r'data\models\ridge_regression.joblib')
#joblib.dump(lasso_model, r'data\models\lasso_regression.joblib')