import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class KANLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Inner function weights (fixed)
        self.inner_kernel = self.add_weight(
            'inner_kernel',
            shape=[input_shape[-1], self.units],
            initializer='glorot_uniform',
            trainable=False
        )
        
        # Outer function weights (adaptive)
        self.outer_kernel = self.add_weight(
            'outer_kernel',
            shape=[self.units, 1],
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.built = True
    
    def call(self, inputs):
        # Inner functions (using periodic functions as per Kolmogorov's theorem)
        inner_outputs = tf.math.sin(tf.matmul(inputs, self.inner_kernel))
        
        # Outer function
        return tf.matmul(inner_outputs, self.outer_kernel)

def create_kan_model(input_dim, units=10):
    inputs = layers.Input(shape=(input_dim,))
    
    # KAN layer
    kan_output = KANLayer(units)(inputs)
    
    # Final output layer
    outputs = layers.Dense(1)(kan_output)
    
    return Model(inputs=inputs, outputs=outputs)

# Example usage
def train_kan_model(X_train, y_train, epochs=100, batch_size=32):
    # Create and compile model
    model = create_kan_model(input_dim=X_train.shape[1])
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    return model, history

# Example data generation for testing
def generate_test_data(n_samples=1000):
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (n_samples, 2))
    y = np.sin(X[:, 0] * np.pi) * np.cos(X[:, 1] * np.pi)
    return X, y.reshape(-1, 1)

#==========================================================
# Generate sample data
X, y = generate_test_data(1000)

# Train the model
model, history = train_kan_model(X, y)

# Make predictions
predictions = model.predict(X)
