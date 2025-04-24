# Next-Generation-Fraud-Detection-Leveraging-Machine-Learning-and-Anomaly-Detection-in-Cybersecurity
# Next-Generation-Fraud-Detection-Leveraging-Machine-Learning-and-Anomaly-Detection-in-Cybersecurity-
 # Random Forest for Fraud Detection

 # random_forest_fraud_detection.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated dataset: Features (Transaction Amount, Frequency, User Activity)
X = [[200, 3, 1], [50, 1, 0], [1000, 10, 3], [15, 0, 0]]  # Example features
y = [0, 1, 0, 1]  # 0 = Non-Fraud, 1 = Fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# K-Means Clustering for Anomaly Detection:

# kmeans_anomaly_detection.py

from sklearn.cluster import KMeans
import numpy as np

# Simulated network traffic data (features: packet size, request frequency)
X = np.array([[10, 200], [12, 150], [15, 300], [100, 5000], [15, 250], [14, 280]])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Predict anomalies (outliers)
predictions = kmeans.predict(X)

# Output clusters
print(f"Cluster Labels: {predictions}")


# Rule-Based Detection System for Suspicious Logins:

# rule_based_login_detection.py

def is_suspicious_login(ip, last_login_time, user_location):
    if user_location != "US" and ip != "trusted_ip":
        return True
    if last_login_time < 1:
        return True
    return False

# Example login attempts
login_attempts = [
    {"ip": "192.168.1.1", "last_login_time": 0.5, "user_location": "Germany"},
    {"ip": "10.10.10.10", "last_login_time": 2, "user_location": "US"}
]

# Check for suspicious logins
for attempt in login_attempts:
    if is_suspicious_login(attempt["ip"], attempt["last_login_time"], attempt["user_location"]):
        print("Suspicious login detected!")


# Support Vector Machine (SVM) for Fraud Classification:

# svm_fraud_detection.py

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Simulated data: Features (Transaction Amount, Account Age, etc.)
X = [[200, 3], [50, 4], [1000, 1], [15, 5]]  # Example features
y = [0, 1, 0, 1]  # 0 = non-fraud, 1 = fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Support Vector Classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predictions and evaluation
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

fraud-detection-project/
├── random_forest_fraud_detection.py
├── kmeans_anomaly_detection.py
├── rule_based_login_detection.py
├── svm_fraud_detection.py
├── README.md
└── requirements.txt
scikit-learn==1.0.2
numpy==1.21.2

# Fraud Detection Project

This project implements several fraud detection techniques using **machine learning** and **anomaly detection**.

## Techniques Implemented:
1. **Random Forest Classifier** - Used for binary classification of fraudulent vs non-fraudulent transactions.
2. **K-Means Clustering** - Detects anomalies in network traffic data.
3. **Rule-Based Detection** - Identifies suspicious login attempts using predefined heuristics.
4. **Support Vector Machine (SVM)** - Classifies fraudulent vs legitimate transactions.

## Prerequisites

- Python 3.6 or higher
- Install the dependencies:


## Running the Scripts
You can run each script individually to test the fraud detection techniques. For example:


## Future Enhancements
- Implement more advanced anomaly detection algorithms like **Isolation Forests** or **Autoencoders**.
- Explore **deep learning** techniques for complex fraud detection scenarios.
- Integrate real-world datasets to validate and improve the models.

# random_forest_fraud_detection.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated dataset: Features (Transaction Amount, Frequency, User Activity)
X = [[200, 3, 1], [50, 1, 0], [1000, 10, 3], [15, 0, 0]]  # Example features
y = [0, 1, 0, 1]  # 0 = Non-Fraud, 1 = Fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# random_forest_fraud_detection.py

# random_forest_fraud_detection.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated dataset: Features (Transaction Amount, Frequency, User Activity)
X = [[200, 3, 1], [50, 1, 0], [1000, 10, 3], [15, 0, 0]]  # Example features
y = [0, 1, 0, 1]  # 0 = Non-Fraud, 1 = Fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predictions and accuracy
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# kmeans_anomaly_detection.py

# kmeans_anomaly_detection.py

from sklearn.cluster import KMeans
import numpy as np

# Simulated network traffic data (features: packet size, request frequency)
X = np.array([[10, 200], [12, 150], [15, 300], [100, 5000], [15, 250], [14, 280]])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Predict anomalies (outliers)
predictions = kmeans.predict(X)

# Output clusters
print(f"Cluster Labels: {predictions}")

# rule_based_login_detection.py

# rule_based_login_detection.py

def is_suspicious_login(ip, last_login_time, user_location):
    if user_location != "US" and ip != "trusted_ip":
        return True
    if last_login_time < 1:
        return True
    return False

# Example login attempts
login_attempts = [
    {"ip": "192.168.1.1", "last_login_time": 0.5, "user_location": "Germany"},
    {"ip": "10.10.10.10", "last_login_time": 2, "user_location": "US"}
]

# Check for suspicious logins
for attempt in login_attempts:
    if is_suspicious_login(attempt["ip"], attempt["last_login_time"], attempt["user_location"]):
        print("Suspicious login detected!")

# svm_fraud_detection.py

# svm_fraud_detection.py

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Simulated data: Features (Transaction Amount, Account Age, etc.)
X = [[200, 3], [50, 4], [1000, 1], [15, 5]]  # Example features
y = [0, 1, 0, 1]  # 0 = non-fraud, 1 = fraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Support Vector Classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predictions and evaluation
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

# requirements.txt
ini
scikit-learn==1.0.2
numpy==1.21.2




