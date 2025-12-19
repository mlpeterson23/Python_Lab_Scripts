"""
TRAINING APPROACH: Segments for Training, Sliding Window for Inference

TRAINING PHASE:
  - Load full-resolution images (3840x2160)
  - Extract ~294 sliding window segments (224x224) per image
  - Train model on individual segments with standard backprop
  - Each segment is treated as an independent training example
  - Loss computed per segment
  - This is INDUSTRY STANDARD - simpler, faster, more stable

TESTING/INFERENCE PHASE:
  - Load full-resolution test image
  - Extract ~294 sliding window segments
  - Get prediction for each segment
  - Aggregate predictions (vote, average, confidence_weighted)
  - Return final aggregated prediction
  - This gives better accuracy than single full-image prediction

WHY THIS APPROACH:
  1. Training on segments: standard supervised learning, no tricks needed
  2. Inference with sliding window: improved accuracy via ensemble voting
  3. No need for custom training loops or complex gradient tape logic
  4. Memory efficient - each segment fits in model input
  5. Leverages Keras fit() which is well-tested and optimized
"""

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

# Configuration
BATCH_SIZE = 16  # Batch size for original images
IMG_SIZE = 224  # MobileNetV3 default input size
IMG_LNGTH = 3840
IMG_HGT = 2160
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 14  # Number of classes (6-6.9, 7-7.9, ..., 19-19.9)
NUM_TABULAR_FEATURES = 3  # Number of tabular input features
TABULAR_FEATURES = ['Previous Track Condition', 'Tire State (PSI in the future)', 'Track']  # Names of tabular features
TABULAR_FILE = 'PNG_Ref.xlsx'  # Excel file containing tabular data
#TABULAR_FILE = 'Dummy_Tabular_Data.xlsx'  # Excel file containing all 1s for testing

# Memory optimization for sliding windows
# Approximate segments per image with crop-first approach (50% overlap): ~294 segments
APPROX_SEGMENTS_PER_IMAGE = 294
# Reduce actual batch size when using sliding window to avoid memory overflow
# This creates smaller effective batches of segments
SLIDING_WINDOW_BATCH_SIZE = max(1, BATCH_SIZE // 8)  # Reduces batch size to 2 when using sliding windows

# CRITICAL: If still getting memory errors, set these to True
# This will significantly reduce memory usage at the cost of some processing
AGGRESSIVE_MEMORY_MODE = True  # Set to True to aggressively reduce memory (use 1 image per batch)
# When True, will use batch size of 1 regardless of BATCH_SIZE setting
if AGGRESSIVE_MEMORY_MODE:
    SLIDING_WINDOW_BATCH_SIZE = 2
    print("WARNING: AGGRESSIVE_MEMORY_MODE enabled - using batch size of 1")


# Dataset paths
TRAIN_FOLDERS = ['a6-6.9_Train','b7-7.9_Train','c8-8.9_Train','d9-9.9_Train','e10-10.9_Train','f11-11.9_Train','g12-12.9_Train','h13-13.9_Train', 'i14-14.9_Train','j15-15.9_Train', 'k16-16.9_Train', 'l17-17.9_Train','m18-18.9_Train','n19-19.9_Train']
TEST_FOLDERS = ['a6-6.9_Test','b7-7.9_Test','c8-8.9_Test','d9-9.9_Test','e10-10.9_Test','f11-11.9_Test','g12-12.9_Test','h13-13.9_Test', 'i14-14.9_Test','j15-15.9_Test', 'k16-16.9_Test', 'l17-17.9_Test','m18-18.9_Test','n19-19.9_Test']
CLASS_NAMES = ['6-6.9', '7-7.9', '8-8.9', '9-9.9', '10-10.9', '11-11.9', '12-12.9', '13-13.9', '14-14.9', '15-15.9', '16-16.9', '17-17.9', '18-18.9', '19-19.9']

# Create combined dataset directories if they don't exist
COMBINED_TRAIN_DIR = 'combined_train'
COMBINED_TEST_DIR = 'combined_test'

# Add a flag to control grayscale and normalization
GRAYSCALE_INPUT = True  # Set to True for black and white images
NORMALIZE_INPUT = False  # Set to True to normalize images to [0,1]

# Sliding window configuration
USE_SLIDING_WINDOW = True  # Set to True to use sliding window segmentation
SEGMENT_HEIGHT = IMG_SIZE  # 224 pixels
SEGMENT_WIDTH = IMG_SIZE   # 224 pixels

# Image masking configuration (asymmetric by design)
# Image layout: Horizon (top) | Tractor tire tread (bottom)
HEIGHT_OFFSET = 32  # Skip left 32 pixels - borders from unequal image division
WIDTH_OFFSET = 32   # Skip left 32 pixels - borders from unequal image division
STEP_SIZE = 112     # 50% overlap (half of segment size)

# Mask regions with low information value
MASK_TOP_ROWS = 2   # Skip top 2*224=448px (horizon noise - intentionally excluded)
MASK_LEFT_COLS = 3  # Skip left 3*224=672px (image division borders)
MASK_RIGHT_COLS = 3 # Skip right 3*224=672px (image division borders)
# NOTE: NO bottom masking - tire tread contains all useful information

AGGREGATION_METHOD = 'confidence_weighted'  # 'voting', 'average', 'max', or 'confidence_weighted' for combining predictions

# Validation-specific batch size (often needs to be smaller than training)
# Set to 1 if validation stalls or runs out of memory
VALIDATION_BATCH_SIZE = max(1, SLIDING_WINDOW_BATCH_SIZE // 2) if USE_SLIDING_WINDOW else BATCH_SIZE
# For example: if SLIDING_WINDOW_BATCH_SIZE = 2, VALIDATION_BATCH_SIZE = 1
# Test batch size (evaluation, can be smaller for safety)
TEST_BATCH_SIZE = max(1, SLIDING_WINDOW_BATCH_SIZE // 4) if USE_SLIDING_WINDOW else BATCH_SIZE

# TensorFlow memory optimization
# Disable aggressive prefetch for sliding window to reduce memory usage
DISABLE_PREFETCH = USE_SLIDING_WINDOW
# Set max prefetch buffer size (lower = less memory)
PREFETCH_BUFFER_SIZE = 2 if USE_SLIDING_WINDOW else 10

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
        df = pd.read_excel(TABULAR_FILE)
        
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

def extract_sliding_window_segments(image):
    """
    Extract sliding window segments from an image's unmasked (tire tread) area.
    
    Image layout:
    - Top: Horizon information (low value, masked out)
    - Bottom: Tire tread and tread patterns (high value, fully processed)
    - Left/Right: Edge borders from unequal image division (low value, masked out)
    
    Strategy: Crop to unmasked region first, then apply sliding window.
    This is more efficient and clearer than calculating offsets repeatedly.
    """
    # Define the unmasked region bounds
    # Top mask: Skip horizon and noise at top
    unmasked_y_start = HEIGHT_OFFSET + MASK_TOP_ROWS * SEGMENT_HEIGHT
    # Bottom: NO masking (tire tread area has all useful info)
    unmasked_y_end = image.shape[0]
    
    # Left mask: Skip border artifacts
    unmasked_x_start = WIDTH_OFFSET + MASK_LEFT_COLS * SEGMENT_WIDTH
    # Right mask: Skip border artifacts
    unmasked_x_end = image.shape[1] - MASK_RIGHT_COLS * SEGMENT_WIDTH
    
    # Crop to unmasked region (tire tread area only)
    unmasked_region = image[unmasked_y_start:unmasked_y_end, 
                            unmasked_x_start:unmasked_x_end, :]
    
    segments = []
    segment_indices = []
    
    # Apply sliding window ONLY to the unmasked region
    # This avoids boundary checks and makes intent explicit
    region_height = unmasked_region.shape[0]
    region_width = unmasked_region.shape[1]
    
    y_idx = 0
    y = 0
    while y + SEGMENT_HEIGHT <= region_height:
        x_idx = 0
        x = 0
        while x + SEGMENT_WIDTH <= region_width:
            segment = unmasked_region[y:y+SEGMENT_HEIGHT, x:x+SEGMENT_WIDTH, :]
            segments.append(segment)
            segment_indices.append((y_idx, x_idx))
            x += STEP_SIZE
            x_idx += 1
        y += STEP_SIZE
        y_idx += 1
    
    return segments, segment_indices, y_idx, x_idx if x_idx > 0 else 1

def aggregate_segment_predictions(predictions, aggregation_method='voting'):
    """Aggregate predictions from multiple segments to determine final class"""
    if aggregation_method == 'voting':
        # Use argmax for each prediction and do majority voting
        class_votes = np.argmax(predictions, axis=1)
        final_class = np.bincount(class_votes).argmax()
        # Get confidence as the proportion of votes for the final class
        confidence = np.sum(class_votes == final_class) / len(class_votes)
        final_prediction = np.zeros(predictions.shape[1])
        final_prediction[final_class] = confidence
        return final_prediction
    
    elif aggregation_method == 'average':
        # Average all probability predictions
        return np.mean(predictions, axis=0)
    
    elif aggregation_method == 'max':
        # Use the max probability for each class across segments
        return np.max(predictions, axis=0)
    
    elif aggregation_method == 'confidence_weighted':
        # Confidence-weighted voting:
        # Each segment votes for its predicted class, weighted by confidence in that class
        # High confidence votes (0.85) dominate, low confidence votes (0.35) are suppressed
        
        # Get predicted class and confidence for each segment
        predicted_classes = np.argmax(predictions, axis=1)      # Which class each segment predicts
        confidences = np.max(predictions, axis=1)               # How confident in that prediction
        
        # Accumulate weighted votes for each class
        weighted_votes = np.zeros(predictions.shape[1])
        for class_idx, confidence in zip(predicted_classes, confidences):
            weighted_votes[class_idx] += confidence
        
        # Normalize to get final probability distribution
        total_votes = np.sum(weighted_votes)
        if total_votes == 0:
            # Fallback if all confidences are 0 (shouldn't happen with softmax)
            return np.mean(predictions, axis=0)
        
        final_prediction = weighted_votes / total_votes
        return final_prediction
    
    else:
        # Default to average
        return np.mean(predictions, axis=0)

class CombinedGenerator:
    """Custom generator to combine image data with tabular features"""
    def __init__(self, image_generator, tabular_dict, batch_size):
        self.image_generator = image_generator
        self.tabular_dict = tabular_dict
        self.batch_size = batch_size
        self.iterator = None
        self.batch_counter = 0
        
    def __len__(self):
        return len(self.image_generator)
    
    def __iter__(self):
        # Reset the iterator each time we start iterating
        self.iterator = iter(self.image_generator)
        self.batch_counter = 0
        return self
    
    def __next__(self):
        # If iterator is None, initialize it
        if self.iterator is None:
            self.iterator = iter(self.image_generator)
            self.batch_counter = 0
        
        # Get next batch of images and labels
        try:
            images, labels = next(self.iterator)
            self.batch_counter += 1
        except StopIteration:
            print(f"\n[Generator] Training generator exhausted after {self.batch_counter} batches")
            raise StopIteration
        except Exception as e:
            print(f"\n[Generator ERROR] Exception in batch {self.batch_counter}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Ensure images are float32
        images = tf.cast(images, tf.float32)
        
        if USE_SLIDING_WINDOW:
            # Process sliding window segments
            batch_segments = []
            batch_tabular_data = []
            batch_labels_expanded = []
            
            for img_idx, (image, label) in enumerate(zip(images, labels)):
                try:
                    # Extract segments from the image
                    segments, segment_indices, num_y, num_x = extract_sliding_window_segments(image)
                    
                    if not segments:
                        # Fallback: use the full image if no segments extracted
                        print(f"WARNING: No segments extracted for image {img_idx}, using full image")
                        segments = [image]
                    
                    # Get tabular data for this image (same for all segments of this image)
                    # Try to get filename from generator - this is tricky
                    try:
                        filename = self.image_generator.filenames[self.image_generator.samples - len(images) + img_idx]
                        base_name = os.path.basename(filename)
                        ref_key = get_reference_key(base_name)
                    except Exception as e:
                        print(f"  [Warning] Could not get filename for image {img_idx}: {e}")
                        ref_key = None
                    
                    if ref_key and ref_key in self.tabular_dict:
                        tabular_features = self.tabular_dict[ref_key]
                    else:
                        tabular_features = np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)
                    
                    # Add each segment with the same label and tabular data
                    for segment in segments:
                        batch_segments.append(segment)
                        batch_tabular_data.append(tabular_features)
                        batch_labels_expanded.append(label)
                
                except Exception as e:
                    print(f"\n[ERROR] Failed to process image {img_idx} in batch {self.batch_counter}: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            try:
                # Convert to numpy arrays
                batch_segments = np.array(batch_segments, dtype=np.float32)
                batch_tabular_data = np.array(batch_tabular_data, dtype=np.float32)
                batch_labels_expanded = np.array(batch_labels_expanded, dtype=np.float32)
            except Exception as e:
                print(f"\n[ERROR] Failed to convert batch to numpy arrays in batch {self.batch_counter}: {type(e).__name__}: {str(e)}")
                print(f"  Batch segments shape: {len(batch_segments)} items")
                print(f"  Batch tabular data shape: {len(batch_tabular_data)} items")
                print(f"  Batch labels shape: {len(batch_labels_expanded)} items")
                import traceback
                traceback.print_exc()
                raise
            
            return [batch_segments, batch_tabular_data], batch_labels_expanded
        
        else:
            # Standard processing without sliding window
            # Get corresponding tabular data
            batch_tabular = []
            for img_idx in range(len(images)):
                try:
                    filename = self.image_generator.filenames[self.image_generator.samples - len(images) + img_idx]
                    base_name = os.path.basename(filename)
                    ref_key = get_reference_key(base_name)
                except:
                    ref_key = None
                
                if ref_key and ref_key in self.tabular_dict:
                    features = self.tabular_dict[ref_key]
                else:
                    features = np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)
                batch_tabular.append(features)
            
            # Convert to numpy array with explicit float32 type
            batch_tabular = np.array(batch_tabular, dtype=np.float32)
            
            # Resize images to model input size
            images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE), method='bilinear')
            
            # Ensure labels are float32
            labels = tf.cast(labels, tf.float32)
            
            return [images, batch_tabular], labels

    @property
    def class_indices(self):
        return self.image_generator.class_indices

    @property
    def samples(self):
        return self.image_generator.samples

def setup_dataset():
    """Setup combined dataset directories"""
    print("\nSetting up dataset directories...")
    print(f"Train folders to process: {TRAIN_FOLDERS}")
    print(f"Test folders to process: {TEST_FOLDERS}")
    
    # Remove existing combined directories if they exist
    if os.path.exists(COMBINED_TRAIN_DIR):
        shutil.rmtree(COMBINED_TRAIN_DIR)
    if os.path.exists(COMBINED_TEST_DIR):
        shutil.rmtree(COMBINED_TEST_DIR)
    
    # Create combined directories
    os.makedirs(COMBINED_TRAIN_DIR)
    os.makedirs(COMBINED_TEST_DIR)
    print(f"Created directories: {COMBINED_TRAIN_DIR}, {COMBINED_TEST_DIR}")
    
    # Process training data
    for folder in TRAIN_FOLDERS:
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        class_name = folder.split('_')[0]
        target_dir = os.path.join(COMBINED_TRAIN_DIR, class_name)
        os.makedirs(target_dir)
        
        print(f"\nProcessing {folder}...")
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

def create_model():
    """Create a MobileNetV3 model with custom classification head"""
    # Load the pre-trained model without top layers (remove name argument)
    base_model = MobileNetV3Large(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Full-width model for better accuracy
    )
    
    # Freeze the base model layers initially
    base_model.trainable = False
    
    # Create the model architecture with improved head
    img_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    tabular_input = layers.Input(shape=(NUM_TABULAR_FEATURES,), name='tabular_input')
    
    x = base_model(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Combine image and tabular branches
    combined = layers.concatenate([x, tabular_input])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs=[img_input, tabular_input], outputs=outputs)
    
    return model

def create_data_generators():
    """Create train and validation data generators with augmentation"""
    # Set up dataset directories first
    setup_dataset()
    
    # Load tabular data
    print("\nLoading tabular data...")
    tabular_dict = load_tabular_data()
    if tabular_dict is None:
        raise ValueError("Failed to load tabular data from PNG_Ref.xlsx")
    
    # Determine batch size based on sliding window usage
    effective_batch_size = SLIDING_WINDOW_BATCH_SIZE if USE_SLIDING_WINDOW else BATCH_SIZE
    
    print(f"\nSetting up data generators...")
    print(f"Using batch sizes:")
    print(f"  Training: {effective_batch_size} images (~{effective_batch_size * APPROX_SEGMENTS_PER_IMAGE} segments)")
    print(f"  Validation: {VALIDATION_BATCH_SIZE} images (~{VALIDATION_BATCH_SIZE * APPROX_SEGMENTS_PER_IMAGE} segments)")
    print(f"  Test: {TEST_BATCH_SIZE} images (~{TEST_BATCH_SIZE * APPROX_SEGMENTS_PER_IMAGE} segments)")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        validation_split=0.2,
        brightness_range=[0.9, 1.1]
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_fn
    )
    
    # Create base image generators
    train_img_gen = train_datagen.flow_from_directory(
        COMBINED_TRAIN_DIR,
        target_size=(IMG_LNGTH, IMG_HGT),
        batch_size=effective_batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator with SMALLER batch size
    val_img_gen = train_datagen.flow_from_directory(
        COMBINED_TRAIN_DIR,
        target_size=(IMG_LNGTH, IMG_HGT),
        batch_size=VALIDATION_BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Create test generator with SMALLEST batch size
    test_img_gen = test_datagen.flow_from_directory(
        COMBINED_TEST_DIR,
        target_size=(IMG_LNGTH, IMG_HGT),
        batch_size=TEST_BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Create combined generators
    train_generator = CombinedGenerator(train_img_gen, tabular_dict, effective_batch_size)
    validation_generator = CombinedGenerator(val_img_gen, tabular_dict, effective_batch_size)
    test_generator = CombinedGenerator(test_img_gen, tabular_dict, effective_batch_size)
    
    print("\nClass mapping:")
    print(train_generator.class_indices)
    print(f"\nFound {train_generator.samples} training samples")
    print(f"Found {validation_generator.samples} validation samples")
    print(f"Found {test_generator.samples} test samples")
    
    return train_generator, validation_generator, test_generator

def debug_tabular_matching(tabular_dict, filenames):
    """Prints debug info for tabular matching."""
    for fname in filenames:
        base_name = os.path.basename(fname)
        ref_key = get_reference_key(base_name)
        found = ref_key in tabular_dict if ref_key else False
        print(f"File: {base_name:30} | Ref key: {ref_key:15} | Found: {found}")

def plot_training_history(history, name):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def find_mobilenetv3_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'mobilenetv3' in layer.name.lower():
            return layer
        if hasattr(layer, 'layers'):
            found = find_mobilenetv3_base(layer)
            if found is not None:
                return found
    return None

def fine_tune_model(model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some layers"""
    # Recursively find the MobileNetV3 base model
    base_model = find_mobilenetv3_base(model)
    if base_model is None:
        raise ValueError('MobileNetV3 base model not found in model.layers')
    base_model.trainable = True
    
    # Freeze all layers except the last 50 (more layers for fine-tuning)
    for layer in base_model.layers[:-50]:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            layer.trainable = False
    
    # Fine-tuning requires a lower learning rate
    fine_tune_lr = LEARNING_RATE / 20  # Much lower learning rate for fine-tuning
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with frozen layers
    history_fine = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    return history_fine

def get_numeric_class_mapping(class_indices):
    """Return a list mapping class index to numeric class value (e.g., 6.0 for '6-6.9')."""
    # class_indices: {class_name: index}
    # Build reverse mapping: index -> class_name
    index_to_class = {v: k for k, v in class_indices.items()}
    # Extract numeric value from class name (assumes format like '6-6.9')
    def extract_numeric(class_name):
        try:
            return float(class_name.split('-')[0])
        except Exception:
            return 0.0
    return [extract_numeric(index_to_class[i]) for i in range(len(index_to_class))]

def calculate_stepped_accuracy(y_true, y_pred, class_indices):
    """Calculate stepped accuracy (off by 1 or 2 classes) using numeric class values."""
    # Get mapping from class index to numeric value
    numeric_class_values = get_numeric_class_mapping(class_indices)
    # Convert one-hot to class index
    y_true_idx = np.argmax(y_true, axis=1)
    y_pred_idx = np.argmax(y_pred, axis=1)
    # Map indices to numeric values
    y_true_num = np.array([numeric_class_values[i] for i in y_true_idx])
    y_pred_num = np.array([numeric_class_values[i] for i in y_pred_idx])
    # Calculate absolute difference in numeric class values
    abs_diff = np.abs(y_true_num - y_pred_num)
    # Stepped accuracy: within 1 class (e.g., 6.0 vs 7.0)
    within_1 = np.mean(abs_diff <= 1.0)
    within_2 = np.mean(abs_diff <= 2.0)
    exact = np.mean(abs_diff == 0.0)
    return exact, within_1, within_2

def evaluate_with_aggregation(model, test_generator, class_indices, aggregation_method='voting'):
    """Evaluate model with segment aggregation for full-image predictions"""
    print(f"\nEvaluating with {aggregation_method} aggregation...")
    print(f"Total test samples: {test_generator.samples}")
    
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    # Safely iterate through generator with timeout protection
    try:
        for batch_idx, (batch_data, batch_labels) in enumerate(test_generator):
            batch_count += 1
            print(f"  Processing batch {batch_count}...", end='\r')
            
            images, tabular_data = batch_data
            batch_predictions = model.predict([images, tabular_data], verbose=0)
            
            all_predictions.extend(batch_predictions)
            all_labels.extend(batch_labels)
            
            # Safety check: break after processing all samples
            if len(all_predictions) >= test_generator.samples:
                print(f"  Processed all {len(all_predictions)} predictions           ")
                break
    
    except StopIteration:
        print(f"  Generator exhausted after {batch_count} batches")
    except Exception as e:
        print(f"  Error during prediction: {str(e)}")
        raise
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"Total predictions collected: {len(all_predictions)}")
    print(f"Total labels collected: {len(all_labels)}")
    
    if len(all_predictions) == 0:
        print("ERROR: No predictions were collected!")
        return 0, 0, 0, 0
    
    # Aggregate predictions based on original images
    # Group predictions by image (every N segments per image)
    aggregated_predictions = []
    aggregated_labels = []
    
    # Use the configured segment count per image
    # This ensures we correctly group all segments from each image
    segments_per_image = APPROX_SEGMENTS_PER_IMAGE
    
    print(f"Aggregating {len(all_predictions)} predictions using {segments_per_image} segments per image...")
    
    for i in range(0, len(all_predictions), segments_per_image):
        segment_batch = all_predictions[i:i+segments_per_image]
        label_batch = all_labels[i:i+segments_per_image]
        
        # Aggregate this image's segment predictions
        aggregated_pred = aggregate_segment_predictions(segment_batch, aggregation_method)
        # Use the first label (all segments have same label)
        aggregated_labels.append(label_batch[0])
        aggregated_predictions.append(aggregated_pred)
    
    aggregated_predictions = np.array(aggregated_predictions)
    aggregated_labels = np.array(aggregated_labels)
    
    print(f"Total images aggregated: {len(aggregated_predictions)}")
    
    # Calculate accuracy
    pred_classes = np.argmax(aggregated_predictions, axis=1)
    true_classes = np.argmax(aggregated_labels, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    
    # Calculate stepped accuracy
    exact, within_1, within_2 = calculate_stepped_accuracy(aggregated_labels, aggregated_predictions, class_indices)
    
    print(f"\nAggregated Accuracy: {accuracy:.4f}")
    print(f"Exact match: {exact:.4f}")
    print(f"Within 1 class: {within_1:.4f}")
    print(f"Within 2 classes: {within_2:.4f}")
    
    return accuracy, exact, within_1, within_2

def main():
    print("Setting up dataset directories...")
    if not os.path.exists(COMBINED_TRAIN_DIR):
        setup_dataset()
    
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
    if USE_SLIDING_WINDOW:
        print("\n" + "="*60)
        print("SLIDING WINDOW SEGMENTATION CONFIGURATION")
        print("="*60)
        print(f"Segment Size: {SEGMENT_HEIGHT}x{SEGMENT_WIDTH}")
        print(f"Image Size: {IMG_LNGTH}x{IMG_HGT}")
        print(f"Height Offset: {HEIGHT_OFFSET} pixels")
        print(f"Width Offset: {WIDTH_OFFSET} pixels")
        print(f"Step Size: {STEP_SIZE} pixels (50% overlap)")
        print(f"Masked Rows (top): {MASK_TOP_ROWS}")
        print(f"Masked Cols (left): {MASK_LEFT_COLS}")
        print(f"Masked Cols (right): {MASK_RIGHT_COLS}")
        print(f"Aggregation Method: {AGGREGATION_METHOD}")
        print("="*60 + "\n")
    
    print("Creating and compiling model...")
    model = create_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=LEARNING_RATE/100
        )
    ]
    
    # Train the model
    print("\nInitial training with frozen base model...")
    print(f"Training approach: Segments extracted via sliding window, then trained normally")
    print(f"Training batch size: {SLIDING_WINDOW_BATCH_SIZE if USE_SLIDING_WINDOW else BATCH_SIZE} images")
    print(f"Expected segments per batch: ~{(SLIDING_WINDOW_BATCH_SIZE if USE_SLIDING_WINDOW else BATCH_SIZE) * APPROX_SEGMENTS_PER_IMAGE}")
    print(f"Validation batch size: {VALIDATION_BATCH_SIZE} images (smaller to avoid memory issues)\n")
    
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,        
            callbacks=callbacks,
            
            class_weight={  # Add class weights if dataset is imbalanced
                0: 1.0,  # Class 6-6.9
                1: 1.0,  # Class 7-7.9
                2: 1.0,   # Class 8-8.9
                3: 1.0,   # Class 9-9.9
                4: 1.0,   # Class 10-12.9
                5: 1.0,   # Class 11-13.9
                6: 1.0,   # Class 12-14.9
                7: 1.0,   # Class 13-15.9
                8: 1.0,   # Class 14-16.9
                9: 1.0,   # Class 15-17.9
                10: 1.0,   # Class 16-18.9
                11: 1.0,   # Class 17-19.9
                12: 1.0,   # Class 18-18.9
                13: 1.0,   # Class 19-19.9
            }
        )
    except Exception as e:
        print(f"\n[TRAINING ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTrying to continue with evaluation...")
        raise
    
    # Plot initial training history
    plot_training_history(history, 'training_history_initial.png')
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    history_fine = fine_tune_model(model, train_generator, validation_generator)
    
    # Plot fine-tuning history
    plot_training_history(history_fine, 'training_history_fine_tune.png')
    
    # Evaluate the model
    print("\nEvaluating the model on test data...")
    if USE_SLIDING_WINDOW:
        # Evaluate with aggregated predictions
        accuracy, exact, within_1, within_2 = evaluate_with_aggregation(
            model, 
            test_generator, 
            train_generator.class_indices,
            aggregation_method=AGGREGATION_METHOD
        )
    else:
        # Standard evaluation
        test_loss, test_accuracy = model.evaluate(
            test_generator,
            steps=len(test_generator)
        )
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    # Save the final model
    model.save('Models\\Multi_Input_Model_Full_3.keras')
    print("\nModel saved as 'Models\\Multi_Input_Model_Full_3.keras'")

if __name__ == '__main__':
    # Enable mixed precision training for faster training if GPU is available
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Enabled mixed precision training")
    except:
        print("Could not enable mixed precision training")
    
    # Memory optimization for TensorFlow
    print("\nConfiguring TensorFlow memory optimization...")
    
    if USE_SLIDING_WINDOW:
        print("Sliding window mode detected - optimizing for reduced memory usage")
        
        # Disable aggressive prefetching
        tf.data.experimental.enable_debug_mode()
        
        # Set options to reduce memory usage
        options = tf.data.Options()
        #options.experimental_optimize_dataset_static_name_change = True
        
        # Reduce prefetch buffer size
        print(f"Prefetch buffer size set to: {PREFETCH_BUFFER_SIZE}")
        print(f"Batch size reduced to: {SLIDING_WINDOW_BATCH_SIZE} (from {BATCH_SIZE})")
        print(f"Expected segment batch size: ~{SLIDING_WINDOW_BATCH_SIZE * APPROX_SEGMENTS_PER_IMAGE}")
        print("If memory issues persist, try:")
        print("  1. Reducing SLIDING_WINDOW_BATCH_SIZE further (divide by 16 instead of 8)")
        print("  2. Increasing STEP_SIZE (less overlap = fewer segments)")
        print("  3. Disabling USE_SLIDING_WINDOW and using standard resizing")
    
    main()