# app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    data_cleaned = data.drop(columns=['Name', 'Surname'])
    X = data_cleaned.drop(columns=['Result'])
    y = data_cleaned['Result']
    return X, y

def train_decision_tree(X, y):
    """Train a Decision Tree model and evaluate its accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

def main():
    """Streamlit app for Decision Tree model training and evaluation."""
    st.title("Decision Tree Model Trainer")

    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

    if uploaded_file is not None:
        st.write("Dataset uploaded successfully!")
        X, y = load_data(uploaded_file)
        st.write("Preview of the dataset:")
        st.dataframe(X.head())

        if st.button("Train Decision Tree Model"):
            with st.spinner("Training the model..."):
                model, accuracy = train_decision_tree(X, y)
            st.success("Model trained successfully!")
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
