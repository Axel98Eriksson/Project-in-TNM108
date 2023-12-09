import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
        
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))

    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
    
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)

gauss_model = make_pipeline(GaussianFeatures(20),LinearRegression())
gauss_model.fit(x[:, np.newaxis], y)
yfit = gauss_model.predict(xfit[:, np.newaxis])


# Load HTML data
data = pd.read_html(r'data\train\strikers.html', header=0, encoding='utf-8', keep_default_na=False)

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
data_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(data_scaled, y, test_size=0.75, random_state=42)

# Linear Regression
linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Ridge Regression
#ridge_model = Ridge(alpha=1.0)  # You can experiment with different alpha values
ridge_model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Lasso Regression
#lasso_model = Lasso(alpha=1.0)  # You can experiment with different alpha values
lasso_model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f'Linear Regression - Mean Squared Error: {mse_linear}')
print(f'Linear Regression - R-squared: {r2_linear}')

# Evaluate Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f'Ridge Regression - Mean Squared Error: {mse_ridge}')
print(f'Ridge Regression - R-squared: {r2_ridge}')

# Evaluate Lasso Regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'Lasso Regression - Mean Squared Error: {mse_lasso}')
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
joblib.dump(linear_model, r'data\models\linear_regression.joblib')
joblib.dump(ridge_model, r'data\models\ridge_regression.joblib')
joblib.dump(lasso_model, r'data\models\lasso_regression.joblib')