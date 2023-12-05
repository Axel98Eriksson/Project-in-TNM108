from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load HTML data
data = pd.read_html(r'data\train\outfield.html', header=0, encoding='utf-8', keep_default_na=False)

# Assuming the first table in the HTML file is the one you want to use
# If there are multiple tables, you may need to inspect and select the correct one
data = data[0]

# Separate player names for visualization
player_names = data['Name']

# Drop player names and info from the features
data = data.drop(['Inf','Name'], axis=1)


y = data['CA']
X = data.drop('CA', axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.2, random_state=42)



# Create and fit the linear regression model
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

line = np.linspace(0,200)

# Scatter plot of actual vs. predicted values
plt.plot(line,line, color = 'red')
plt.scatter(y_test, y_pred, alpha=0.2)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Regression line
plt.scatter(X_test[:, 0], y_test , color='black', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='blue', linewidth=3, label='Predicted')
plt.title('Regression Line')
plt.xlabel('Feature 1')
plt.ylabel('Target Variable')
plt.legend()
plt.show()
