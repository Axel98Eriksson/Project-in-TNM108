import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

data = pd.read_html(r'data\season0\DC.html', header=0, encoding='utf-8', keep_default_na=False)
v0 = joblib.load(r'data\models\v02.joblib')
data = data[0]

player_names = data['Name']
CA = data['CA']
data = data.drop(['Inf', 'Name', 'CA'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(data['Position'])
data['Position'] = labelEncoder.transform(data['Position'])
labelEncoder.fit(data['Right Foot'])
data['Right Foot'] = labelEncoder.transform(data['Right Foot'])
labelEncoder.fit(data['Left Foot'])
data['Left Foot'] = labelEncoder.transform(data['Left Foot'])


scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
potential_signings = v0.predict(scaled_data)

#potential_signings = v0.predict(data)
potential_signings = pd.DataFrame(potential_signings, columns=['CA'])
result = pd.concat([player_names, potential_signings], axis=1)
result['Actual CA'] = CA
result['CA diff'] = abs(result['Actual CA']-result['CA'])

result_sorted = result.sort_values(by='CA diff', ascending=False)

# Display the top 5 rows
top_targets = result_sorted.head(100)
print(top_targets)