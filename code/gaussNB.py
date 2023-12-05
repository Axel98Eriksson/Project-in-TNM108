import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have a dataset with player statistics and labels (position and role)
# Replace 'your_dataset.csv' with the actual dataset file path
generalattackers = pd.read_html(r"data\train\attackers-general.html", header=0, encoding="utf-8", keep_default_na=False)
TFs= pd.read_html(r"data\train\tf-s.html", header=0, encoding="utf-8", keep_default_na=False)

potential_recruits_html = pd.read_html(r"data\test\attackers.html", header=0, encoding="utf-8", keep_default_na=False)
potential_recruits = potential_recruits_html[0]


df_tf = TFs[0]
df_tf['is_targeted_position'] = 1

df_ga = generalattackers[0]
df_ga['is_targeted_position'] = 0

data_dirty = pd.concat([df_ga, df_tf], ignore_index=True)
data = data_dirty.drop(['Inf',	'Name',	'Position',	'Nat',	'Age',	'Club',	'Transfer Value',	'Wage',	'Min AP',	'Min Fee Rls',	'Min Fee Rls to Foreign Clubs','Personality',	'Media Handling',	'Left Foot',	'Right Foot', 'Height', 'CA'],axis=1)

# Assuming you have columns like 'Position', 'Role', and various player statistics
# Modify this based on your actual dataset columns
#features = data.drop(['Position', 'Role', 'CurrentAbility'], axis=1)  # Exclude 'CurrentAbility' from features
#labels = data['Role']  # Now predicting player roles

# Filter data to include only attacking players
#attacking_players_data = data[data['Position'].str.contains('Attack')]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['is_targeted_position'], axis=1),
    data['is_targeted_position'],
    test_size=0.2,
    random_state=42
)

# Create and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
predictions = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Now, let's evaluate potential recruits
# Assuming you have a DataFrame called 'potential_recruits' with their statistics
# Replace this with the actual data of potential recruits
recruits_predictions = gnb.predict()

# Find the best players suited for the target forward role on support duty
best_players_indices = [i for i, target in enumerate(recruits_predictions) if target == 1]
best_players = potential_recruits.iloc[best_players_indices]

# Print the best players
print("Best Players Suited for Target Forward (Support) Role:")
print(best_players)
