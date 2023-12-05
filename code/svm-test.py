import pandas as pd
import glob
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
df_ga['Recruitment_Status'] = 0

df = pd.concat([df_ga, df_tf], ignore_index=True)
df_cleaned = df.drop(['Inf',	'Name',	'Position',	'Nat',	'Age',	'Club',	'Transfer Value',	'Wage',	'Min AP',	'Min Fee Rls',	'Min Fee Rls to Foreign Clubs','Personality',	'Media Handling',	'Left Foot',	'Right Foot', 'Height', 'CA'],axis=1)



#=======================SVM==================================
# Load your dataset (replace 'your_dataset.csv' with your actual file)


# Assuming your dataset has a column 'Recruitment_Status' indicating the target variable
X = df_cleaned.drop('Recruitment_Status', axis=1)  # Features
y = df_cleaned['Recruitment_Status']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize/Normalize the features (optional but often recommended for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model (you can experiment with different kernels and hyperparameters)
svm_model = SVC(kernel='linear', C=1.0)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5, scoring='accuracy')

# Print the cross-validation scores
print("Cross-validation Scores:", cv_scores)
print("Mean Accuracy: ", cv_scores.mean())

targeted_position_rows = df[df_cleaned['Recruitment_Status'] == 1]

print(targeted_position_rows)