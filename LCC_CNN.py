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
import math


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
SLIDING_WINDOW_BATCH_SIZE = 128 

# CRITICAL: If still getting memory errors, set these to True
# This will significantly reduce memory usage at the cost of some processing
AGGRESSIVE_MEMORY_MODE = False  # Set to True to aggressively reduce memory (use 1 image per batch)
# When True, will use batch size of 1 regardless of BATCH_SIZE setting
if AGGRESSIVE_MEMORY_MODE:
    SLIDING_WINDOW_BATCH_SIZE = 1
    print("WARNING: AGGRESSIVE_MEMORY_MODE enabled - using batch size of 1")
IS_DATA_SETUP = True #To avoid resetting data if not needed

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
VALIDATION_BATCH_SIZE = BATCH_SIZE
# For example: if SLIDING_WINDOW_BATCH_SIZE = 2, VALIDATION_BATCH_SIZE = 1
# Test batch size (evaluation, can be smaller for safety)
TEST_BATCH_SIZE = max(1, SLIDING_WINDOW_BATCH_SIZE // 4) if USE_SLIDING_WINDOW else BATCH_SIZE

# TensorFlow memory optimization
# Disable aggressive prefetch for sliding window to reduce memory usage
DISABLE_PREFETCH = USE_SLIDING_WINDOW
# Set max prefetch buffer size (lower = less memory)
PREFETCH_BUFFER_SIZE = 2 if USE_SLIDING_WINDOW else 10

# LCC modifications
DATA_LOCATION = os.path.join(os.environ["SCRATCH"], "LCC_UP") + "/" # Path to data location (CHANGE AS NEEDED)

def get_reference_key(filename):
    """Extract reference key from filename based on naming convention"""
    if filename[0].isdigit():
        # For files starting with number, get everything before 'f'
        match = re.match(r'([^f]+)', filename)
        if match:
            return match.group(1)
    else:
        # For files starting with letter, get everything in between first '_' and before '_seg###'
        parts = filename.split('_')
        Final_parts = parts[1].rsplit('_seg')
        if len(parts) > 1:
            return Final_parts[0]
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

def setup_dataset():
    """Setup combined dataset directories"""
    print("\nSetting up dataset directories...")
    print(f"Train folders to process: {TRAIN_FOLDERS}")
    print(f"Test folders to process: {TEST_FOLDERS}")
    
    # Remove existing combined directories if they exist
    if os.path.exists(DATA_LOCATION + COMBINED_TRAIN_DIR):
        shutil.rmtree(DATA_LOCATION + COMBINED_TRAIN_DIR)
    if os.path.exists(DATA_LOCATION + COMBINED_TEST_DIR):
        shutil.rmtree(DATA_LOCATION + COMBINED_TEST_DIR)
    
    # Create combined directories
    os.makedirs(DATA_LOCATION + COMBINED_TRAIN_DIR)
    LCC_COMBINED_TRAIN_DIR = DATA_LOCATION + COMBINED_TRAIN_DIR
    os.makedirs(DATA_LOCATION + COMBINED_TEST_DIR)
    print(f"Created directories: {COMBINED_TRAIN_DIR}, {COMBINED_TEST_DIR}")
    LCC_COMBINED_TEST_DIR = DATA_LOCATION + COMBINED_TEST_DIR
    
    # Process training data
    for folder in TRAIN_FOLDERS:
        if not os.path.exists(DATA_LOCATION + folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        class_name = folder.split('_')[0]
        target_dir = os.path.join(DATA_LOCATION + COMBINED_TRAIN_DIR, class_name)
        os.makedirs(target_dir)
        
        print(f"\nProcessing {folder}...")
        files = [f for f in os.listdir(DATA_LOCATION + folder) if f.lower().endswith('.png')]
        if not files:
            print(f"No PNG files found in {folder}")
            continue
        
        # Copy PNG files
        total_segments = 0
        for file in files:
            seg_num = 0
            source_file = os.path.join(DATA_LOCATION + folder, file)
            #Segment Images and save to target directory
            Processed_image = preprocess_input_fn(tf.keras.preprocessing.image.load_img(source_file))
            segments, segment_indices, y_idx, x_idx = extract_sliding_window_segments(Processed_image)
            for segment in segments:
                segment_filename = f"{os.path.splitext(file)[0]}_seg{seg_num}.png"
                target_file = os.path.join(target_dir, segment_filename)
                tf.keras.preprocessing.image.save_img(target_file, segment)
                seg_num += 1
            total_segments = total_segments + len(segments)
        print(f"Copied {total_segments} segments from {folder}")
    
    # Process test data
    for folder in TEST_FOLDERS:
        if not os.path.exists(DATA_LOCATION + folder):
            print(f"Warning: {folder} not found, skipping...")
            continue
        
        class_name = folder.split('_')[0]
        target_dir = os.path.join(DATA_LOCATION + COMBINED_TEST_DIR, class_name)
        os.makedirs(target_dir)
        
        print(f"\nProcessing test folder {folder}...")
        files = [f for f in os.listdir(DATA_LOCATION + folder) if f.lower().endswith('.png')]
        if not files:
            print(f"No PNG files found in {folder}")
            continue
        
        # Copy PNG files
        for file in files:
            source_file = os.path.join(DATA_LOCATION + folder, file)
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

def load_image_and_tabular(image_path, label, tabular_dict):
    """
    Load image and corresponding tabular data (tf.data compatible).
    Note: Must work in tf.data.map() graph mode - no .numpy() calls!
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    
    # Resize to model input size
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method='bilinear')
    
    # Grayscale conversion if needed
    if GRAYSCALE_INPUT:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
    
    # Extract filename for tabular data lookup
    # Split path to get filename (works in graph mode)
    filename_tensor = tf.strings.split(image_path, sep='/')[-1]
    
    # Use py_function to call Python code (extract ref key and lookup)
    # This is safe because it's only called during data loading, not forward pass
    def get_tabular_features(filename_tensor):
        # Convert tensor to string for processing
        filename = filename_tensor.numpy().decode('utf-8')
        ref_key = get_reference_key(filename)
        
        if ref_key and ref_key in tabular_dict:
            return tabular_dict[ref_key].astype(np.float32)
        else:
            return np.zeros(NUM_TABULAR_FEATURES, dtype=np.float32)
    
    # Call py_function to get tabular features
    tabular_features = tf.py_function(
        func=get_tabular_features,
        inp=[filename_tensor],
        Tout=tf.float32
    )
    
    # Set shape explicitly for tf.data optimization
    tabular_features.set_shape([NUM_TABULAR_FEATURES])
    
    return (image, tabular_features), label


def create_optimized_dataset(image_paths, labels, tabular_dict, batch_size, 
                            augment=False, shuffle=True):
    """
    Create an optimized tf.data pipeline with parallel loading and prefetching.
    This replaces the slow custom generator and provides 80-95% GPU utilization.
    
    Uses tf.keras.layers for augmentation (built-in, graph-mode compatible).
    """
    # Create dataset from image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        # Shuffle with a buffer - larger buffer = more randomness but more memory
        buffer_size = min(5000, max(1000, len(image_paths) // 2))
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    # Load and preprocess images in parallel on CPU cores
    dataset = dataset.map(
        lambda path, label: load_image_and_tabular(path, label, tabular_dict),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch the data BEFORE augmentation (augmentation works on batches)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Convert integer labels to one-hot encoding AFTER batching
    # This ensures labels have shape (batch_size, NUM_CLASSES)
    def one_hot_encode(batch_images_tabular, batch_labels):
        batch_labels_onehot = tf.one_hot(batch_labels, NUM_CLASSES, dtype=tf.float32)
        return batch_images_tabular, batch_labels_onehot
    
    dataset = dataset.map(
        one_hot_encode,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if training (using built-in Keras layers - graph mode safe)
    if augment:
        # Create augmentation pipeline using Keras layers
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),  # 50% chance of horizontal flip
            layers.RandomFlip("vertical"),    # 50% chance of vertical flip
            layers.RandomBrightness(0.1),     # Random brightness adjustment ±10%
        ])
        
        # Apply augmentation to the batched dataset
        # Note: Augmentation only affects the image, not the tabular features or labels
        def unpack_and_augment(batch_images_tabular, batch_labels_onehot):
            batch_images = batch_images_tabular[0]  # Images from the tuple
            batch_tabular = batch_images_tabular[1]  # Tabular features from the tuple
            
            # Apply augmentation to images
            augmented_images = data_augmentation(batch_images, training=True)
            
            return (augmented_images, batch_tabular), batch_labels_onehot
        
        dataset = dataset.map(
            unpack_and_augment,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Prefetch to keep GPU fed with data during training
    # AUTOTUNE determines optimal prefetch buffer size automatically
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_data_generators():
    """Create train, validation, and test datasets using optimized tf.data pipeline"""
    # Load tabular data
    print("\nLoading tabular data...")
    tabular_dict = load_tabular_data()
    if tabular_dict is None:
        raise ValueError("Failed to load tabular data from PNG_Ref.xlsx")
    
    # Determine batch size based on sliding window usage
    effective_batch_size = SLIDING_WINDOW_BATCH_SIZE if USE_SLIDING_WINDOW else BATCH_SIZE
    
    print(f"\nSetting up optimized tf.data pipeline...")
    LCC_COMBINED_TRAIN_DIR = DATA_LOCATION + COMBINED_TRAIN_DIR
    LCC_COMBINED_TEST_DIR = DATA_LOCATION + COMBINED_TEST_DIR
    
    # Helper function to collect image paths and labels from directory structure
    def get_image_paths_and_labels(directory):
        """Walk directory and collect image paths with class labels"""
        image_paths = []
        labels = []
        class_to_idx = {}
        
        for class_idx, class_name in enumerate(sorted(os.listdir(directory))):
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            class_to_idx[class_name] = class_idx
            for image_name in os.listdir(class_dir):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, image_name)
                    image_paths.append(image_path)
                    labels.append(class_idx)
        
        return np.array(image_paths, dtype=object), np.array(labels, dtype=np.int32), class_to_idx
    
    print("  Collecting training image paths...")
    train_paths, train_labels, class_to_idx = get_image_paths_and_labels(LCC_COMBINED_TRAIN_DIR)
    
    print("  Collecting test image paths...")
    test_paths, test_labels, _ = get_image_paths_and_labels(LCC_COMBINED_TEST_DIR)
    
    # Split training into train and validation (80/20 split)
    train_size = int(0.8 * len(train_paths))
    indices = np.random.permutation(len(train_paths))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_paths_split = train_paths[train_indices]
    train_labels_split = train_labels[train_indices]
    
    val_paths_split = train_paths[val_indices]
    val_labels_split = train_labels[val_indices]
    
    # Create optimized datasets
    print("  Creating training dataset...")
    train_dataset = create_optimized_dataset(
        train_paths_split, train_labels_split, tabular_dict, 
        effective_batch_size, augment=True, shuffle=True
    )
    
    print("  Creating validation dataset...")
    val_dataset = create_optimized_dataset(
        val_paths_split, val_labels_split, tabular_dict,
        VALIDATION_BATCH_SIZE, augment=False, shuffle=False
    )
    
    print("  Creating test dataset...")
    test_dataset = create_optimized_dataset(
        test_paths, test_labels, tabular_dict,
        TEST_BATCH_SIZE, augment=False, shuffle=False
    )
    
    # Store metadata for later use (aggregation, evaluation)
    # model.fit() works with tf.data.Dataset directly
    class DatasetInfo:
        """Store metadata about dataset without wrapping the dataset itself"""
        def __init__(self, dataset, num_samples, class_indices):
            self.dataset = dataset
            self.samples = num_samples
            self.class_indices = class_indices
    
    train_generator = DatasetInfo(train_dataset, len(train_labels_split), class_to_idx)
    validation_generator = DatasetInfo(val_dataset, len(val_labels_split), class_to_idx)
    test_generator = DatasetInfo(test_dataset, len(test_labels), class_to_idx)

    print("\nClass mapping:")
    print(train_generator.class_indices)
    print(f"\nFound {train_generator.samples} training samples")
    print(f"Found {validation_generator.samples} validation samples")
    print(f"Found {test_generator.samples} test samples")
    print(f"\n✓ Data pipeline optimized!")
    print(f"  - Parallel image loading with AUTOTUNE")
    print(f"  - AUTOTUNE prefetching to GPU")
    print(f"  - Expected GPU utilization: 80-95%")
    
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

def evaluate_with_aggregation(model, test_generator_info, class_indices, aggregation_method='voting'):
    """Evaluate model with segment aggregation for full-image predictions"""
    test_generator = test_generator_info.dataset
    
    print(f"\nEvaluating with {aggregation_method} aggregation...")
    
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    # Safely iterate through generator with timeout protection
    try:
        for batch_idx, ((images, tabular_data), batch_labels) in enumerate(test_generator):
            batch_count += 1
            print(f"  Processing batch {batch_count}...", end='\r')
            
            # Make predictions
            batch_predictions = model.predict([images, tabular_data], verbose=0)
            
            # Convert to numpy if needed
            all_predictions.append(batch_predictions)
            all_labels.append(batch_labels.numpy() if hasattr(batch_labels, 'numpy') else batch_labels)
            
            # Safety check: break after processing all samples (Removed, re-add later. Ref error with .samples)
            #if len(all_predictions) * batch_predictions.shape[0] >= test_generator_info.samples:
                #print(f"  Processed all samples           ")
                #break
    
    except StopIteration:
        print(f"  Generator exhausted after {batch_count} batches")
    except Exception as e:
        print(f"  Error during prediction: {str(e)}")
        raise
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
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
    if not os.path.exists(DATA_LOCATION + COMBINED_TRAIN_DIR):
        setup_dataset()
    
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Determine batch size based on sliding window usage
    effective_batch_size = SLIDING_WINDOW_BATCH_SIZE if USE_SLIDING_WINDOW else BATCH_SIZE
    
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
            filepath=os.path.join(DATA_LOCATION, 'best_model.keras'),
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
    print(f"Training approach: Optimized tf.data pipeline with {effective_batch_size} batch size")
    print(f"GPU-optimized data loading with AUTOTUNE prefetching")
    print(f"Mixed precision (float16) enabled\n")
    
    try:
        # Pass tf.data.Dataset directly to model.fit()
        # No need to specify steps_per_epoch - it auto-calculates from dataset
        history = model.fit(
            train_generator.dataset,  # Use the underlying tf.data.Dataset
            epochs=EPOCHS,
            validation_data=validation_generator.dataset,  # Use the underlying tf.data.Dataset
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
        raise
    
    # Plot initial training history
    plot_training_history(history, 'training_history_initial.png')
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    history_fine = fine_tune_model(model, train_generator.dataset, validation_generator.dataset)
    
    # Plot fine-tuning history
    plot_training_history(history_fine, 'training_history_fine_tune.png')
    
    # Evaluate the model
    print("\nEvaluating the model on test data...")
    if USE_SLIDING_WINDOW:
        # Evaluate with aggregated predictions
        accuracy, exact, within_1, within_2 = evaluate_with_aggregation(
            model, 
            test_generator,  # Pass DatasetInfo object
            train_generator.class_indices,
            aggregation_method=AGGREGATION_METHOD
        )
    else:
        # Standard evaluation
        test_loss, test_accuracy = model.evaluate(
            test_generator.dataset  # Use the underlying dataset
        )
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    
    # Save the final model
    model.save(DATA_LOCATION + 'Models\\Sliding_Window_RSTL_Test.keras')
    print("\nModel saved as 'Sliding_Window_RSTL_Test.keras'")

if __name__ == '__main__':
    # GPU Configuration - DO THIS FIRST
    print("Configuring GPU...")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Detected GPUs: {gpus}")
    
    if gpus:
        try:
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")
    
    # Enable mixed precision training for faster training if GPU is available
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Enabled mixed precision training (float16)")
    except Exception as e:
        print(f"Could not enable mixed precision training: {e}")
    
    # TensorFlow graph optimization
    print("Enabling TensorFlow optimizations...")
    tf.config.run_functions_eagerly(False)  # Use graph mode (faster)
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": True,
        "function_optimization": True,
        "arithmetic_optimization": True,
    })
    print("✓ Graph optimizations enabled")
    
    # Memory optimization for TensorFlow
    
    main()