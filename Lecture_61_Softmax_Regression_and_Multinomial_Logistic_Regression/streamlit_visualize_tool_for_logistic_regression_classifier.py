'''
This Python file is a **Streamlit-based interactive visualization tool for Logistic Regression classifiers**. 

It allows users to experiment with logistic regression on synthetic datasets and visually explore how different hyperparameters affect the model's decision boundaries and accuracy.

**Key Features:**

- **Dataset Selection:** Users can choose between a binary or multiclass synthetic dataset, which is generated and plotted automatically.

- **Hyperparameter Controls:** The sidebar provides interactive controls for regularization type (`penalty`), regularization strength (`C`), solver, maximum iterations, multi-class strategy, and `l1_ratio` (for elasticnet).

- **Visualization:** The tool displays the dataset and, after running the algorithm, overlays the decision boundary learned by the logistic regression model.

- **Model Training and Evaluation:** When the "Run Algorithm" button is clicked, the model is trained on the selected dataset and parameters. The tool then shows the updated plot and displays the model's accuracy on the test set.

**Purpose:**  

This app is designed for educational and exploratory purposes, helping users understand how logistic regression works, 

how its parameters influence classification, and how decision boundaries change in response to different settings—all in an interactive, visual way.

'''


# Import necessary libraries
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Function to load and plot the initial dataset (binary or multiclass)
def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        # Generate a binary classification dataset
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        # Generate a multiclass classification dataset
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y


# Function to create a meshgrid for plotting decision boundaries
def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array


# Set the matplotlib style for better visuals
plt.style.use('fivethirtyeight')


# Streamlit sidebar title
st.sidebar.markdown("# Logistic Regression Classifier")


# Sidebar: Dataset selection (Binary or Multiclass)
dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)


# Sidebar: Regularization type selection
penalty = st.sidebar.selectbox(
    'Regularization',
    ('l2', 'l1', 'elasticnet', 'none')
)


# Sidebar: Regularization strength (C) input
c_input = float(st.sidebar.number_input('C', value=1.0))


# Sidebar: Solver selection for logistic regression
solver = st.sidebar.selectbox(
    'Solver',
    ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
)


# Sidebar: Maximum number of iterations input
max_iter = int(st.sidebar.number_input('Max Iterations', value=100))


# Sidebar: Multi-class strategy selection
multi_class = st.sidebar.selectbox(
    'Multi Class',
    ('auto', 'ovr', 'multinomial')
)


# Sidebar: l1_ratio input (for elasticnet)
l1_ratio = int(st.sidebar.number_input('l1 Ratio'))


# Create a matplotlib figure and axis for plotting
fig, ax = plt.subplots()


# Load and plot the initial dataset
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)  # Display the initial plot in Streamlit


# If the user clicks the "Run Algorithm" button
if st.sidebar.button('Run Algorithm'):
    orig.empty()  # Clear the previous plot

    # Create and train the logistic regression model with selected parameters
    clf = LogisticRegression(
        penalty=penalty,
        C=c_input,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        l1_ratio=l1_ratio
    )
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Generate meshgrid and predict labels for decision boundary
    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    # Plot the decision boundary
    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)  # Display the updated plot

    # Show the accuracy score in Streamlit
    st.subheader("Accuracy for Logistic Regression  " + str(round(accuracy_score(y_test, y_pred), 2)))