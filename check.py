import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys

def load_category_data(category_name):
    """Load the .npy file for a specific category."""
    try:
        # Look for the file in the quickdraw_data directory
        data_dir = "quickdraw_data"
        # Find the first file that matches the category name
        for filename in os.listdir(data_dir):
            if category_name.lower() in filename.lower() and filename.endswith('.npy'):
                file_path = os.path.join(data_dir, filename)
                print(f"Loading data from {file_path}")
                return np.load(file_path)
        
        # If no file was found
        print(f"No data file found for category: {category_name}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_all_categories():
    """Load all category names from the quickdraw_data directory."""
    categories = []
    data_dir = "quickdraw_data"
    
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.npy'):
                # Extract category name from filename
                category = filename.replace('.npy', '').replace('full_numpy_bitmap_', '')
                categories.append(category)
        return sorted(categories)
    except Exception as e:
        print(f"Error loading categories: {e}")
        return []

def preprocess_image(image):
    """Preprocess the image to match the model's expected input."""
    # Reshape to match the model's input shape (assuming 28x28)
    image = image.reshape(1, 28, 28, 1)
    # Normalize pixel values
    image = image / 255.0
    return image

def get_predictions(model, image, categories):
    """Get the top 5 predictions for the given image."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Get the indices of the top 5 predictions
    top_indices = np.argsort(predictions[0])[-5:][::-1]
    
    # Get the corresponding class names and probabilities
    results = []
    for idx in top_indices:
        if idx < len(categories):
            category = categories[idx]
            probability = predictions[0][idx] * 100
            results.append((category, probability))
    
    return results

def display_image_and_predictions(image, predictions):
    """Display the image and its top 5 predictions."""
    plt.figure(figsize=(8, 6))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title('Input Drawing')
    plt.axis('off')
    
    # Display the predictions
    plt.subplot(1, 2, 2)
    categories = [p[0] for p in predictions]
    probabilities = [p[1] for p in predictions]
    
    y_pos = np.arange(len(categories))
    
    plt.barh(y_pos, probabilities, align='center')
    plt.yticks(y_pos, categories)
    plt.xlabel('Probability (%)')
    plt.title('Top 5 Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the model
    try:
        model = load_model('quickdraw_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get all available categories
    all_categories = load_all_categories()
    if not all_categories:
        print("No categories found. Please check the data directory.")
        return
    
    # Get user input
    print("Available categories:")
    print(", ".join(all_categories))
    
    category_name = input("Enter category name: ")
    
    # Load the data for the selected category
    category_data = load_category_data(category_name)
    if category_data is None:
        return
    
    # Get index from user
    try:
        max_index = len(category_data) - 1
        print(f"This category has {len(category_data)} drawings (indices 0 to {max_index})")
        index = int(input(f"Enter index (0-{max_index}): "))
        
        if index < 0 or index > max_index:
            print(f"Index out of range. Please enter a value between 0 and {max_index}.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return
    
    # Get the specified drawing
    image = category_data[index]
    
    # Get and display predictions
    predictions = get_predictions(model, image, all_categories)
    
    # Print top 5 predictions
    print("\nTop 5 Predictions:")
    for i, (category, probability) in enumerate(predictions, 1):
        print(f"{i}. {category}: {probability:.2f}%")
    
    # Display the image and predictions graphically
    display_image_and_predictions(image, predictions)

if __name__ == "__main__":
    main()