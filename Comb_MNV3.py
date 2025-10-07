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
BATCH_SIZE = 16
IMG_SIZE = 224  # MobileNetV3 default input size
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 14  # Number of classes (6-6.9, 7-7.9, ..., 19-19.9)
NUM_TABULAR_FEATURES = 3  # Number of tabular input features
TABULAR_FEATURES = ['Previous Track Condition', 'Tire State (PSI in the future)', 'Track']  # Names of tabular features

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
    
    print("\nSetting up data generators...")
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
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    val_img_gen = train_datagen.flow_from_directory(
        COMBINED_TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Create test generator
    test_img_gen = test_datagen.flow_from_directory(
        COMBINED_TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Create combined generators
    train_generator = CombinedGenerator(train_img_gen, tabular_dict, BATCH_SIZE)
    validation_generator = CombinedGenerator(val_img_gen, tabular_dict, BATCH_SIZE)
    test_generator = CombinedGenerator(test_img_gen, tabular_dict, BATCH_SIZE)
    
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

def plot_training_history(history):
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
    plt.savefig('training_history.png')
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

def main():
    print("Setting up dataset directories...")
    if not os.path.exists(COMBINED_TRAIN_DIR):
        setup_dataset()
    
    print("Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
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
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,        
        callbacks=callbacks,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        
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
    
    # Plot initial training history
    plot_training_history(history)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    history_fine = fine_tune_model(model, train_generator, validation_generator)
    
    # Plot fine-tuning history
    plot_training_history(history_fine)
    
    # Evaluate the model
    print("\nEvaluating the model on test data...")
    test_loss, test_accuracy = model.evaluate(
        test_generator,
        steps=len(test_generator)
    )
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the final model
    model.save('Models\\Multi_Input_Model_Full_3.keras')
    print("\nModel saved as 'Multi_Input_Model_Full_3.keras'")

if __name__ == '__main__':
    # Enable mixed precision training for faster training if GPU is available
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Enabled mixed precision training")
    except:
        print("Could not enable mixed precision training")
    
    main()