import tensorflow as tf
import pathlib
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import shutil
import re
from sklearn.metrics import confusion_matrix, classification_report, r2_score
import seaborn as sns

# Configuration
BATCH_SIZE = 16
IMG_SIZE = 224  # MobileNetV3 default input size
EPOCHS = 100
NUM_CLASSES = 14  # Number of classes (6-6.9, 7-7.9, ..., 19-19.9)
NUM_TABULAR_FEATURES = 3  # Number of tabular input features
TABULAR_FEATURES = ['Previous Track Condition', 'Tire State (PSI in the future)', 'Track']  # Names of tabular features
TEST_FOLDERS = ['a6-6.9_Test','b7-7.9_Test','c8-8.9_Test','d9-9.9_Test','e10-10.9_Test','f11-11.9_Test','g12-12.9_Test','h13-13.9_Test', 'i14-14.9_Test','j15-15.9_Test', 'k16-16.9_Test', 'l17-17.9_Test','m18-18.9_Test','n19-19.9_Test']
CLASS_NAMES = ['6-6.9', '7-7.9', '8-8.9', '9-9.9', '10-10.9', '11-11.9', '12-12.9', '13-13.9', '14-14.9', '15-15.9', '16-16.9', '17-17.9', '18-18.9', '19-19.9']
COMBINED_TEST_DIR = 'combined_test'
GRAYSCALE_INPUT = True  # Use grayscale input
NORMALIZE_INPUT = False  # Normalize input images

# Import Trained Model
def load_model(model_path):
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
def get_reference_key(filename):
    """Extract reference key from filename based on naming convention"""
    if filename[0].isdigit():
        # For files starting with number, get everything before 'f'
        match = re.match(r'([^f]+)', filename)
        if match:
            return match.group(1)
    else:
        # For files starting with letter, get everything after first '_'
        parts = filename.split('_')
        if len(parts) > 1:
            return parts[1]
    return None
    
def load_tabular_data():
    """Load and process tabular data from Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel('PNG_Ref.xlsx')
        
        # Create dictionary mapping reference keys to tabular features
        tabular_dict = {}
        for _, row in df.iterrows():
            ref_key = str(row.iloc[0])  # First column as reference
            # Explicitly convert features to float32
            features = row.iloc[1:4].astype('float32').values  # Columns 2, 3, and 4 as features
            tabular_dict[ref_key] = features
            
        return tabular_dict
    except Exception as e:
        print(f"Error loading tabular data: {str(e)}")
        return None

class CombinedGenerator:
    """Custom generator to combine image data with tabular features"""
    def __init__(self, image_generator, tabular_dict, batch_size):
        self.image_generator = image_generator
        self.tabular_dict = tabular_dict
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.image_generator)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get next batch of images and labels
        images, labels = next(self.image_generator)
        
        # Ensure images are float32
        images = tf.cast(images, tf.float32)
        
        # Get corresponding tabular data
        batch_tabular = []
        for filename in self.image_generator.filenames[-len(images):]:
            base_name = os.path.basename(filename)
            ref_key = get_reference_key(base_name)
            if ref_key and ref_key in self.tabular_dict:
                # Tabular dict values are already float32 from load_tabular_data
                features = self.tabular_dict[ref_key]
            else:
                # Use zeros if no matching tabular data found
                features = np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)
            batch_tabular.append(features)
        
        # Convert to numpy array with explicit float32 type
        batch_tabular = np.array(batch_tabular, dtype=np.float32)
        
        # Ensure labels are float32
        labels = tf.cast(labels, tf.float32)
        
        return [images, batch_tabular], labels
    
def setup_dataset():
    """Setup combined dataset directories"""
    print(f"Test folders to process: {TEST_FOLDERS}")
       # Remove existing combined directories if they exist
    if os.path.exists(COMBINED_TEST_DIR):
        shutil.rmtree(COMBINED_TEST_DIR)
    
    # Create combined directories
    os.makedirs(COMBINED_TEST_DIR)
    print(f"Created directory: {COMBINED_TEST_DIR}")
    # Process test data
    for folder in TEST_FOLDERS:
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        class_name = folder.split('_')[0]
        target_dir = os.path.join(COMBINED_TEST_DIR, class_name)
        os.makedirs(target_dir)
        
        print(f"\nProcessing test folder {folder}...")
        files = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
        if not files:
            print(f"No PNG files found in {folder}")
            continue
        
        # Copy PNG files
        for file in files:
            source_file = os.path.join(folder, file)
            target_file = os.path.join(target_dir, file)
            shutil.copy2(source_file, target_file)
        print(f"Copied {len(files)} images from {folder}")

def preprocess_input_fn(img):
    """Preprocess image input with explicit data type handling"""
    # Convert to float32 first for consistent calculations
    img = tf.cast(img, tf.float32)
    
    # Convert to grayscale if needed
    if GRAYSCALE_INPUT:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)  # Keep 3 channels for MobileNetV3
    
    # Normalize if needed
    if NORMALIZE_INPUT:
        img = img / 255.0
    
    return img

class CombinedGenerator:
    """Custom generator to combine image data with tabular features"""
    def __init__(self, image_generator, tabular_dict, batch_size):
        self.image_generator = image_generator
        self.tabular_dict = tabular_dict
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.image_generator)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Get next batch of images and labels
        images, labels = next(self.image_generator)
        
        # Ensure images are float32
        images = tf.cast(images, tf.float32)
        
        # Get corresponding tabular data
        batch_tabular = []
        for filename in self.image_generator.filenames[-len(images):]:
            base_name = os.path.basename(filename)
            ref_key = get_reference_key(base_name)
            if ref_key and ref_key in self.tabular_dict:
                # Tabular dict values are already float32 from load_tabular_data
                features = self.tabular_dict[ref_key]
            else:
                # Use zeros if no matching tabular data found
                features = np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)
            batch_tabular.append(features)
        
        # Convert to numpy array with explicit float32 type
        batch_tabular = np.array(batch_tabular, dtype=np.float32)
        
        # Ensure labels are float32
        labels = tf.cast(labels, tf.float32)
        
        return [images, batch_tabular], labels
    @property
    def class_indices(self):
        return self.image_generator.class_indices

    @property
    def samples(self):
        return self.image_generator.samples

def create_data_generators():
    """Create train and validation data generators with augmentation"""
    # Set up dataset directories first
    setup_dataset()
    
    # Load tabular data
    print("\nLoading tabular data...")
    tabular_dict = load_tabular_data()
    if tabular_dict is None:
        raise ValueError("Failed to load tabular data from PNG_Ref.xlsx")
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_fn
    )
    test_img_gen = test_datagen.flow_from_directory(
        COMBINED_TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = CombinedGenerator(test_img_gen, tabular_dict, BATCH_SIZE)
    print(f"Found {test_generator.samples} test samples")
    return test_generator

def calculate_stepped_accuracy(y_true_classes, y_pred_classes, class_names):
    """Calculate accuracy metrics with steps (exact, ±1 class, ±2 classes)"""
    total = len(y_true_classes)
    exact_matches = np.sum(y_true_classes == y_pred_classes)
    
    # Calculate one-off accuracy
    off_by_one = 0
    off_by_two = 0
    
    for true, pred in zip(y_true_classes, y_pred_classes):
        if true != pred:
            diff = abs(true - pred)
            if diff == 1:
                off_by_one += 1
            elif diff == 2:
                off_by_two += 1
    
    # Calculate percentages
    exact_acc = exact_matches / total * 100
    within_one_acc = (exact_matches + off_by_one) / total * 100
    within_two_acc = (exact_matches + off_by_one + off_by_two) / total * 100
    
    print("\nStepped Accuracy Metrics:")
    print(f"Exact matches: {exact_matches}/{total} ({exact_acc:.2f}%)")
    print(f"Off by 1 class: {off_by_one}/{total} ({off_by_one/total*100:.2f}%)")
    print(f"Off by 2 classes: {off_by_two}/{total} ({off_by_two/total*100:.2f}%)")
    print(f"Within ±1 class accuracy: {within_one_acc:.2f}%")
    print(f"Within ±2 class accuracy: {within_two_acc:.2f}%")
    
    # Create a more detailed breakdown
    print("\nPrediction Distance Distribution:")
    max_diff = max(abs(y_true_classes - y_pred_classes))
    for diff in range(max_diff + 1):
        count = np.sum(abs(y_true_classes - y_pred_classes) == diff)
        print(f"Off by {diff} classes: {count}/{total} ({count/total*100:.2f}%)")

def get_ordered_class_names(generator):
    """Return class names ordered by the generator's class_indices mapping."""
    class_indices = generator.class_indices
    ordered_class_names = [None] * len(class_indices)
    for folder, idx in class_indices.items():
        # Remove '_Test' from folder name for label
        label = folder.replace('_Test', '')
        ordered_class_names[idx] = label
    return ordered_class_names

def main():
    print("Setting up dataset directories...")
    if not os.path.exists(COMBINED_TEST_DIR):
        setup_dataset()
    
    # Load the trained model
    model_path = 'Models\\Multi_Input_Model_Full_0.keras'
    model = load_model(model_path)
    
    test_generator = create_data_generators()
    
    # Evaluate the model
    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        steps=len(test_generator)
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # --- Confusion Matrix and R-squared Calculation ---
    # Collect all test data and predictions
    all_labels = []
    all_predictions = []
    test_generator.image_generator.reset()
    for _ in range(len(test_generator)):
        batch_x, batch_y = next(test_generator)
        batch_pred = model.predict(batch_x, verbose=0)
        all_labels.append(batch_y)
        all_predictions.append(batch_pred)
    y_true = np.concatenate(all_labels)
    y_pred_proba = np.concatenate(all_predictions)
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)
    
    # Get class names in the correct order for plotting
    ordered_class_names = get_ordered_class_names(test_generator.image_generator)
    print("Class indices mapping:", test_generator.image_generator.class_indices)

    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(ticks=np.arange(NUM_CLASSES) + 0.5, labels=CLASS_NAMES, rotation=45)
    plt.yticks(ticks=np.arange(NUM_CLASSES) + 0.5, labels=CLASS_NAMES, rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_Full_0.png')
    plt.close()
    print('Confusion matrix saved as confusion_matrix_Full_0.png')
    
    # Calculate stepped accuracy metrics
    calculate_stepped_accuracy(y_true_classes, y_pred_classes, ordered_class_names)
    
    # Classification Report
    print('\nClassification Report:')
    print(classification_report(y_true_classes, y_pred_classes, target_names=ordered_class_names))
    
    # R-squared Calculation (one-vs-rest for each class)
    r2_scores = []
    for i in range(y_pred_proba.shape[1]):
        r2 = r2_score(y_true[:, i], y_pred_proba[:, i])
        r2_scores.append(r2)
    print('\nR-squared values for each class:')
    for i, r2 in enumerate(r2_scores):
        print(f'Class {i}: {r2:.4f}')
    print(f'Average R-squared: {np.mean(r2_scores):.4f}')

    # Save model predictions
    predictions_df = pd.DataFrame({
        'True Label': y_true_classes,
        'Predicted Label': y_pred_classes
    })
    predictions_df.to_excel('model_predictions.xlsx', index=False)
    print('Model predictions saved to model_predictions.xlsx')

    # Calculate stepped accuracy metrics
    calculate_stepped_accuracy(y_true_classes, y_pred_classes, ordered_class_names)

if __name__ == "__main__":
    main()



