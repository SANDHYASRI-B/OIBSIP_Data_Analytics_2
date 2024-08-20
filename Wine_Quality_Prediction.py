import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv('/content/WineQT.csv')

# Display first few rows of the dataset
df.head()
# Basic data information
print("Shape of the Dataframe: ",df.shape)
print()

print("Information about Dataframe: ")
print(df.info())

print()
print("Description of the Dataframe: ")
print(df.describe())

print()
# checking for missing values
print("Sum of missing/null values: ")
print(df.isnull().sum())

correlation = df.corr()
# constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt = '.1f', annot = True, annot_kws={'size':7}, cmap = 'Blues')

threshold = 6
df['quality_binary'] = df['quality'].apply(lambda x: 'Good' if x > threshold else 'Bad')
X = df.drop(['quality', 'quality_binary'], axis=1)
Y = df['quality_binary']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=45)
print(model)
# Train the model
model.fit(X_train, Y_train)
# Make predictions
Y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

plt.figure(figsize=(4, 4))
sns.countplot(x='quality_binary', data=df,color="teal", width=0.5)
plt.show()
