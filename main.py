import argparse
import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Function to create different neural network architectures
def create_model(architecture):
    if architecture == '1':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    elif architecture == '2':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='linear'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='linear'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='linear'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    elif architecture == '3':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    # Add more architecture options as needed

    return model


# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()


# Function to plot training and validation accuracy
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()


# Function to plot training and validation loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Neural network for breast cancer classification')
parser.add_argument('--architecture', type=str, default='1',
                    help='Select the neural network architecture (1, 2, 3)')
args = parser.parse_args()

# Generate breast cancer dataset using make_blobs
data, labels = make_blobs(n_samples=1000, centers=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the selected neural network model
model = create_model(args.architecture)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on test data
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate performance using relevant metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
plot_confusion_matrix(y_test, y_pred)

# Plot training and validation accuracy
plot_accuracy(history)

# Plot training and validation loss
plot_loss(history)
