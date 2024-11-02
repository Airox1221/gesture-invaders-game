import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class DecisionTreeModel:
    def __init__(self, data_file, model_file='decision_tree_model.joblib'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.class_names = None
        # Define the specific features used in the model
        self.features = ['x5', 'y5', 'x8', 'y8', 'x9', 'y9', 'x12', 'y12', 'x13', 'y13', 'x16', 'y16', 'x17', 'y17', 'x20', 'y20']

    def load_data(self):
        """Load the dataset and separate it into features and target variable."""
        try:
            data = pd.read_csv(self.data_file)
            X = data[self.features]
            y = data['class']
            return X, y
        except FileNotFoundError:
            print(f"Error: The file '{self.data_file}' was not found.")
            return None, None

    def train(self):
        """Train the Decision Tree model using the provided dataset."""
        print("Training strated")
        X, y = self.load_data()
        if X is None or y is None:
            return

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Define class names based on unique labels in `y`
        self.class_names = [str(c) for c in np.unique(y)]

        # Initialize and train the Decision Tree Classifier
        self.model = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # Save the trained model
        self.save_model()

    def save_model(self):
        """Save the trained model to a file."""
        if self.model:
            joblib.dump((self.model, self.class_names, self.features), self.model_file)
            print(f"Model saved as '{self.model_file}'")
        else:
            print("No model to save. Train the model first.")

    def load_model(self):
        """Load the trained model from a file."""
        try:
            self.model, self.class_names, self.features = joblib.load(self.model_file)
            print("Model loaded successfully")
        except FileNotFoundError:
            print(f"Model file '{self.model_file}' not found. Train and save the model first.")

    def predict(self, sample_data):
        """Make predictions using the trained model."""
        if self.model:
            # Ensure sample_data is in the correct shape (1, 16 features)
            if len(sample_data) != len(self.features):
                print(f"Error: Expected {len(self.features)} features, but got {len(sample_data)}")
                return 2

            prediction = self.model.predict([sample_data])
            # print("Sample Prediction:", prediction)
            return int(prediction[0])
        else:
            print("Model not loaded. Load the model first.")
            return 2

    def visualize_tree(self):
        """Visualize the Decision Tree."""
        if self.model:
            plt.figure(figsize=(12, 8))
            tree.plot_tree(self.model, feature_names=self.features, class_names=self.class_names, filled=True)
            plt.show()
        else:
            print("Model not loaded. Load the model first.")

# Example usage
'''
if __name__ == "__main__":
    # Initialize the DecisionTreeModel with data file
    dt_model = DecisionTreeModel(data_file='finger_nodes.csv')

    # Train and save the model
    # Uncomment the next line to train the model
    # dt_model.train()

    # Load the model
    dt_model.load_model()

    # Predict with a sample data
    sample_data = [228,307,220,353,186,329,176,370,142,351,136,378,98,371,99,385]  # Replace with actual values
    dt_model.predict(sample_data)

    # Visualize the decision tree
    # Uncomment the next line to visualize the tree
    # dt_model.visualize_tree()
    '''
