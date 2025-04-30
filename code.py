import pandas as pd

# Load the dataset
data = pd.read_csv('dataset.csv')

data.head()

data.isnull()

# import pandas_profiling
# profile = pandas_profiling.ProfileReport(data)
# profile.to_file("eda_report.html")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Split the data into features and target variable
x = data.drop('RiskLevel', axis=1)  # Replace 'target_column' with the actual column name you want to predict
y = data['RiskLevel']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier()
}

results = {}
for clf_name, clf in classifiers.items():
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label='positive', average='macro' )
    precision = precision_score(y_test, y_pred, pos_label='positive', average='macro')
    recall = recall_score(y_test, y_pred, pos_label='positive', average='macro')
    results[clf_name] = {'Accuracy': accuracy, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}

    print(f'{clf_name} - Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

### Step 4: Create Comparison Charts

### python
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame from the results
results_df = pd.DataFrame(results).T
# Plotting using Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='Accuracy', data=results_df)
plt.title('Accuracy Comparison')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='F1 Score', data=results_df)
plt.title('F1 Score Comparison')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='Precision', data=results_df)
plt.title('Precision Comparison')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y='Recall', data=results_df)
plt.title('Recall Comparison')
plt.xticks(rotation=45)
plt.show()

plt.hist(x)
sns.pairplot(x)
