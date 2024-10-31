import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
data_dir = r"D:\dataset\Iris.csv"  # Sử dụng chuỗi raw
iris_data = pd.read_csv(data_dir)

# Separate features and target variable
X = iris_data.drop(columns=['Species'])
y = iris_data['Species']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize classifiers
svm_classifier = SVC(kernel='linear')
dt_classifier = DecisionTreeClassifier()
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train classifiers
svm_classifier.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

# Make predictions
svm_predictions = svm_classifier.predict(X_test)
dt_predictions = dt_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)

# Create a scatter plot for two features
plt.figure(figsize=(15, 10))

# Choose two features for the plot (e.g., SepalLengthCm and SepalWidthCm)
feature1 = 'SepalLengthCm'
feature2 = 'SepalWidthCm'

# Function to plot predictions
def plot_predictions(X, predictions, title):
    plt.scatter(X[feature1], X[feature2], c=predictions, cmap='viridis', edgecolor='k', s=100)
    plt.title(title)

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)

# Plot SVM predictions
plt.subplot(1, 3, 1)
plot_predictions(X_test, svm_predictions, 'SVM Predictions')

# Plot Decision Tree predictions
plt.subplot(1, 3, 2)
plot_predictions(X_test, dt_predictions, 'Decision Tree Predictions')

# Plot KNN predictions
plt.subplot(1, 3, 3)
plot_predictions(X_test, knn_predictions, 'KNN Predictions')

# Show the plots
plt.tight_layout()
plt.show()