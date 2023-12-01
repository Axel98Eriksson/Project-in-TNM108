import pandas as pd
import glob
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# finds most recent file in specified folder
#list_of_files = glob.glob(os.path.join(r'data\train', '*'))
#latest_file = max(list_of_files, key=os.path.getctime)
#print(latest_file)

generalattackers = pd.read_html(r"data\train\attackers-general.html", header=0, encoding="utf-8", keep_default_na=False)
TFs= pd.read_html(r"data\train\tf-s.html", header=0, encoding="utf-8", keep_default_na=False)

#create dataframe with general attackers
df_tf = TFs[0]
df_tf['is_targeted_position'] = 1


df_ga = generalattackers[0]
df_ga['is_targeted_position'] = 0

df = pd.concat([df_ga, df_tf], ignore_index=True)
df_cleaned = df.drop(['Inf',	'Name',	'Position',	'Nat',	'Age',	'Club',	'Transfer Value',	'Wage',	'Min AP',	'Min Fee Rls',	'Min Fee Rls to Foreign Clubs','Personality',	'Media Handling',	'Left Foot',	'Right Foot', 'Height'],axis=1)

y = df_cleaned['is_targeted_position']
X = df_cleaned.drop('is_targeted_position', axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print(X_test)

#=======================SVM==================================
# Initialize the SVM model (you can experiment with different kernels and hyperparameters)
#svm_model = SVC(kernel='linear', C=1.0)

# Train the SVM model
#svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
#y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
#accuracy = accuracy_score(y_test, y_pred)
#report = classification_report(y_test, y_pred)

#print(f'Accuracy: {accuracy}')
#print('Classification Report:')
#print(report)
