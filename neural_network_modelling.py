# Import library
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

# Read data for modelling
df = pd.read_csv(
    "clean_data.csv",
    sep="\t",
)
df = df.drop('Unnamed: 0', axis=1)

# Data modelling will use 'clean text' column
data_preprocessed = df['clean text'].tolist()

# Feature extraction
count_vect = CountVectorizer()
count_vect.fit(data_preprocessed)

X = count_vect.transform(data_preprocessed)
print("Feature Extraction selesai")

# Save feature extraction result
pickle.dump(count_vect, open("feature.p", "wb"))

# Split dataset into training and test data
classes = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size = 0.2)

# Model training with neural network method
model = MLPClassifier()
model.fit(X_train, y_train)
print("Train selesai")

# Save data modelling result
pickle.dump(model, open("model.p", "wb"))

# Model testing
test = model.predict(X_test)
print("Testing selesai")
print(classification_report(y_test, test))

# Cross validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)

accuracies = []

y = classes

for iteration, data in enumerate(kf.split(X), start=1):
    data_train = X[data[0]]
    target_train = y[data[0]]
    
    data_test = X[data[1]]
    target_test = y[data[1]]
    
    clf = MLPClassifier()
    clf.fit(data_train, target_train)
    
    preds = clf.predict(data_test)
    
    accuracy = accuracy_score(target_test, preds)
    
    print("Training ke-", iteration)
    print(classification_report(target_test,preds))
    print("==================================================")
    
    accuracies.append(accuracy)
    
average_accuracy = np.mean(accuracies)

print()
print()
print()
print("Rata-rata Accuracy: ", average_accuracy)

# Save data modelling result
pickle.dump(model, open("model.p", "wb"))
print("Model saved!")