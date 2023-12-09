# Name: Mukelo Sanele Ziyane
# Student ID: 09892054

# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the dataset
url = "./winequality-white.csv"
df = pd.read_csv(url, sep=";")

# Set page layout to wide for a better user interface
st.set_page_config(layout="wide")

# Sidebar for exploration and model selection
st.sidebar.title("Explore the Dataset and Choose Model")
if st.sidebar.checkbox("Show Raw Data"):
    st.sidebar.write(df)

st.sidebar.header("Choose Classification Model")
model_option = st.sidebar.selectbox(
    "Select a model", ("Random Forest", "Support Vector Machine", "K-Nearest Neighbors")
)

# Display dataset statistics
st.sidebar.header("Dataset Statistics")
st.sidebar.write("Number of samples:", df.shape[0])
st.sidebar.write(
    "Number of features:", df.shape[1] - 1
)  # excluding the target 'quality'
st.sidebar.write("Number of classes:", df["quality"].nunique())
st.sidebar.write("Class Distribution:")
st.sidebar.write(df["quality"].value_counts())

# Split the data into features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train models
if model_option == "Random Forest":
    model = RandomForestClassifier()
elif model_option == "Support Vector Machine":
    model = make_pipeline(StandardScaler(), SVC())
elif model_option == "K-Nearest Neighbors":
    model = KNeighborsClassifier()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display classification results
st.subheader("Classification Results")
st.write("Predicted Quality:", y_pred)

# Display model evaluation results
st.subheader("Model Evaluation Results")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.write(f"Accuracy: {accuracy:.2%}")
st.write(f"Precision: {precision:.2%}")
st.write(f"F1 Score: {f1:.2%}")

# Display confusion matrix
st.subheader("Confusion Matrix")
conf_matrix = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt.gcf())  # Explicitly pass the Matplotlib figure

# Streamlit App for user input features
if st.sidebar.checkbox("Predict with User Input"):
    st.title("White Wine Classification App")

    # Sidebar for user input features
    st.sidebar.header("User Input Features")

    # Add widgets for user input features
    fixed_acidity = st.sidebar.slider(
        "Fixed Acidity",
        float(df["fixed acidity"].min()),
        float(df["fixed acidity"].max()),
        float(df["fixed acidity"].mean()),
    )
    volatile_acidity = st.sidebar.slider(
        "Volatile Acidity",
        float(df["volatile acidity"].min()),
        float(df["volatile acidity"].max()),
        float(df["volatile acidity"].mean()),
    )
    citric_acid = st.sidebar.slider(
        "Citric Acid",
        float(df["citric acid"].min()),
        float(df["citric acid"].max()),
        float(df["citric acid"].mean()),
    )
    residual_sugar = st.sidebar.slider(
        "Residual Sugar",
        float(df["residual sugar"].min()),
        float(df["residual sugar"].max()),
        float(df["residual sugar"].mean()),
    )
    chlorides = st.sidebar.slider(
        "Chlorides",
        float(df["chlorides"].min()),
        float(df["chlorides"].max()),
        float(df["chlorides"].mean()),
    )
    free_sulfur_dioxide = st.sidebar.slider(
        "Free Sulfur Dioxide",
        float(df["free sulfur dioxide"].min()),
        float(df["free sulfur dioxide"].max()),
        float(df["free sulfur dioxide"].mean()),
    )
    total_sulfur_dioxide = st.sidebar.slider(
        "Total Sulfur Dioxide",
        float(df["total sulfur dioxide"].min()),
        float(df["total sulfur dioxide"].max()),
        float(df["total sulfur dioxide"].mean()),
    )
    density = st.sidebar.slider(
        "Density",
        float(df["density"].min()),
        float(df["density"].max()),
        float(df["density"].mean()),
    )
    pH = st.sidebar.slider(
        "pH", float(df["pH"].min()), float(df["pH"].max()), float(df["pH"].mean())
    )
    sulphates = st.sidebar.slider(
        "Sulphates",
        float(df["sulphates"].min()),
        float(df["sulphates"].max()),
        float(df["sulphates"].mean()),
    )
    alcohol = st.sidebar.slider(
        "Alcohol",
        float(df["alcohol"].min()),
        float(df["alcohol"].max()),
        float(df["alcohol"].mean()),
    )

    # Combine user inputs into a DataFrame
    user_inputs = pd.DataFrame(
        {
            "Fixed Acidity": [fixed_acidity],
            "Volatile Acidity": [volatile_acidity],
            "Citric Acid": [citric_acid],
            "Residual Sugar": [residual_sugar],
            "Chlorides": [chlorides],
            "Free Sulfur Dioxide": [free_sulfur_dioxide],
            "Total Sulfur Dioxide": [total_sulfur_dioxide],
            "Density": [density],
            "pH": [pH],
            "Sulphates": [sulphates],
            "Alcohol": [alcohol],
        }
    )

    # Display user input features
    st.write("### User Input Features")
    st.write(user_inputs)

    # Display prediction
    st.write("### Prediction")
    # Add a button to trigger predictions
    if st.button("Predict"):
        prediction = model.predict(user_inputs)
        st.success(f"Predicted Quality: {prediction[0]}")

    # Display model evaluation metrics
    st.write("### Model Evaluation Metrics")
    # Display accuracy, precision, and f1 score
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Precision: {precision:.2%}")
    st.write(f"F1 Score: {f1:.2%}")
