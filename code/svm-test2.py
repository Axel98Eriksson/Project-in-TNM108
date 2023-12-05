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
df_tf['Recruitment_Status'] = 1

df_ga = generalattackers[0]
df_ga['Recruitment_Status'] = 0

df = pd.concat([df_ga, df_tf], ignore_index=True)
df_cleaned = df.drop(['Inf',	'Name',	'Position',	'Nat',	'Age',	'Club',	'Transfer Value',	'Wage',	'Min AP',	'Min Fee Rls',	'Min Fee Rls to Foreign Clubs','Personality',	'Media Handling',	'Left Foot',	'Right Foot', 'Height', 'CA'],axis=1)



#=======================SVM==================================
# Load your dataset (replace 'your_dataset.csv' with your actual file)

# ... (previous code)

# Create a copy of the dataframe to avoid modifying the original data
df_combined = df.copy()

# Assuming that best target forwards are marked somehow in your original data
# Let's say they have a 'Best_Forward' column with a value of 1
# Adjust this condition based on your actual data
df_combined['Player_Type'] = df_combined['Best_Forward'].apply(lambda x: 1 if x == 1 else 0)

# Assuming that players open to joining are marked somehow in your original data
# Let's say they have an 'Open_to_Join' column with a value of 1
# Adjust this condition based on your actual data
df_combined['Open_to_Join'] = df_combined['Open_to_Join_Column'].apply(lambda x: 1 if x == 1 else 0)

# Separate features and target variable
X_combined = df_combined.drop(['Recruitment_Status', 'Player_Type', 'Open_to_Join'], axis=1)
y_combined = df_combined['Recruitment_Status']

# Split the data into training and testing sets
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Standardize/Normalize the features (optional but often recommended for SVM)
scaler_combined = StandardScaler()
X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)
X_test_combined_scaled = scaler_combined.transform(X_test_combined)

# Initialize the SVM model (you can experiment with different kernels and hyperparameters)
svm_model_combined = SVC(kernel='linear', C=1.0)

# Train the SVM model
svm_model_combined.fit(X_train_combined_scaled, y_train_combined)

# Predict recruitment status for all players
all_players_combined_scaled = scaler_combined.transform(X_combined)
all_players_combined_predictions = svm_model_combined.predict(all_players_combined_scaled)

# Add predicted recruitment status to the DataFrame
df_combined['Predicted_Recruitment_Status'] = all_players_combined_predictions

# Rank players based on predicted probabilities
df_combined_ranked = df_combined.sort_values(by='Predicted_Recruitment_Status', ascending=False)

# Display the top-ranked players who are open to joining
top_players_open_to_join = df_combined_ranked[df_combined_ranked['Open_to_Join'] == 1].head()

print("Top-ranked players open to joining:")
print(top_players_open_to_join)

# ... (any other analysis or output you need)

