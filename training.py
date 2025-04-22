import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import os
import glob

from tqdm import tqdm

# 1. Data loading function (memory-efficient)
def data_generator(data_dir, categories, batch_size=32, samples_per_class=10000):
    """Generate batches of data for training"""
    while True:
        # Randomly select categories for this batch
        batch_categories = np.random.choice(categories, size=batch_size, replace=True)
        
        X_batch = []
        y_batch = []
        
        # Load samples for each category in batch
        for category_name in batch_categories:
            # Get the label index for this category
            label_idx = categories.index(category_name)
            
            # Load the numpy file if not already in memory
            file_path = os.path.join(data_dir, f"{category_name}.npy")
            data = np.load(file_path)
            
            # Select a random sample
            random_idx = np.random.randint(0, len(data))
            image = data[random_idx].reshape(28, 28, 1)
            
            X_batch.append(image)
            y_batch.append(label_idx)
        
        # Convert to numpy arrays
        X_batch = np.array(X_batch, dtype=np.float32) / 255.0  # Normalize
        y_batch = np.array(y_batch)
        
        yield X_batch, tf.keras.utils.to_categorical(y_batch, num_classes=len(categories))

# 2. Build the CNN model
def build_model(num_classes):
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. Training setup
data_dir = "quickdraw_data"
categories = [os.path.basename(f).replace('.npy', '') for f in glob.glob(os.path.join(data_dir, "*.npy"))]
print(f"Found {len(categories)} categories")

# Create validation data
val_images = []
val_labels = []

# Create a small validation set by sampling from each category
for idx, category in enumerate(tqdm(categories)):
    file_path = os.path.join(data_dir, f"{category}.npy")
    data = np.load(file_path)
    
    # Get 100 samples for validation
    val_samples = data[:100]
    val_images.append(val_samples)
    val_labels.append(np.full(len(val_samples), idx))

# Combine validation data
val_images = np.vstack(val_images).reshape(-1, 28, 28, 1)
val_labels = np.concatenate(val_labels)
val_images = val_images.astype(np.float32) / 255.0
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=len(categories))

# 4. Create and train the model
model = build_model(num_classes=len(categories))
print(model.summary())

# Enable mixed precision for faster training
mixed_precision_policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)

# Train using the generator
steps_per_epoch = len(categories) * 5  # Arbitrary number based on your needs
train_generator = data_generator(data_dir, categories)

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('quickdraw_model_best.h5', save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00001)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=(val_images, val_labels),
    callbacks=callbacks
)

# 5. Save the model for later use in your app
model.save('quickdraw_model.h5')

# Optional: Convert to TensorFlow Lite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TF Lite model
with open('quickdraw_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete and model saved!")