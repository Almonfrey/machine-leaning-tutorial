import hyperopt
from hyperopt import hp, fmin, tpe, Trials, early_stop
import numpy as np
import wandb

import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from sklearn.model_selection import train_test_split

# Creates a tf.Dataset generator
def create_tf_dataset(X, y, batch_size):

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(seed=42, buffer_size=len(X)).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

# Train the model
def train_model(params):
    
    project_name = 'hyperopt_mnist'
    # wandb log loop execution. Fold level logging
    wandb.init(project=project_name)
    wandb_callback = wandb.keras.WandbCallback(save_model=False)

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Split train to have the validation set
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # put data in a tf.Dataset pipeline
    train_data = create_tf_dataset(X_train, y_train, params['batch_size'])
    eval_data = create_tf_dataset(X_eval, y_eval, params['batch_size'])

    # Build MLP model with given hyperparameters
    model = Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        Dense(
            params['layer_sizes'], 
            activation=params['activation']
        ),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=params['learn_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=params['patience'],
    verbose=1
    )

    # Train the model 
    history = model.fit(
        train_data,
        epochs=params['epochs'],
        validation_data=eval_data,
        verbose=1,
        callbacks=[early_stopping, wandb_callback]
    )

    # Get the last iteration validation loss as hyperopt score
    eval_score = history.history["val_loss"][-1]
    
    return {
        'loss': eval_score,
        'status': hyperopt.STATUS_OK,
    }

# Objective function to minimize (negative accuracy)
def objective(params):
    return train_model(params)

# Define search space for hyperparameters
space = {
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'epochs': hp.choice('epochs', [5, 10, 20]),
    'layer_sizes': hp.choice('layer_sizes', [64, 128, 256]),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'learn_rate': hp.loguniform('learn_rate', -5, 0),
    'patience': hp.choice('early_stopping_patience', [3, 5, 7])
}

# Perform hyperparameter optimization
trials = Trials()
best = fmin(fn=objective, space=space, 
            algo=tpe.suggest, max_evals=50, 
            trials=trials, rstate=np.random.default_rng(42),
            early_stop_fn=early_stop.no_progress_loss(5)
)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(hyperopt.space_eval(space, best))