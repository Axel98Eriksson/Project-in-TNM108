import pandas as pd
import glob
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# finds most recent file in specified folder
list_of_files = glob.glob(os.path.join(r'data\train', '*'))
latest_file = max(list_of_files, key=os.path.getctime)
print(latest_file)

list = pd.read_html(latest_file, header=0, encoding="utf-8", keep_default_na=False)

#create dataframe from list
df = list[0]
df['is_targeted_position'] = 1
df.add(df)

print(df[["Name","CA"]].sort_values(by = 'CA',ascending=False))

X = df.drop('is_targeted_position', axis=1)
y = df['is_targeted_position']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Gaussian Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)








