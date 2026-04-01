import gc
import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV3Large
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Configuration - Import from main file or define here
IMG_SIZE = 224
IMG_LNGTH = 3840
IMG_HGT = 2160
NUM_CLASSES = 14
SEGMENTS_PER_IMAGE = 294
# MobileNetV3Large output dimension after GlobalAveragePooling2D (used for feature pre-extraction)
FEATURE_DIM = 960
GRAYSCALE_INPUT = True
SEGMENT_HEIGHT = IMG_SIZE
SEGMENT_WIDTH = IMG_SIZE
HEIGHT_OFFSET = 32
WIDTH_OFFSET = 32
STEP_SIZE = 112
MASK_TOP_ROWS = 2
MASK_LEFT_COLS = 3
MASK_RIGHT_COLS = 3
EPOCHS = 100
CLASS_NAMES = ['6-6.9', '7-7.9', '8-8.9', '9-9.9', '10-10.9', '11-11.9', '12-12.9', '13-13.9', '14-14.9', '15-15.9', '16-16.9', '17-17.9', '18-18.9', '19-19.9']
K_CROSS_FOLDS = 5
DATA_LOCATION = os.path.join(os.environ["SCRATCH"], "LCC_UP") + "/"  # Set to your data location if using external path
COMBINED_DIR = os.path.join(DATA_LOCATION, 'combined_data_directory')
TRAIN_FOLDERS = ['a6-6.9_Train','b7-7.9_Train','c8-8.9_Train','d9-9.9_Train','e10-10.9_Train','f11-11.9_Train','g12-12.9_Train','h13-13.9_Train', 'i14-14.9_Train','j15-15.9_Train', 'k16-16.9_Train', 'l17-17.9_Train','m18-18.9_Train','n19-19.9_Train']
TEST_FOLDERS = ['a6-6.9_Test','b7-7.9_Test','c8-8.9_Test','d9-9.9_Test','e10-10.9_Test','f11-11.9_Test','g12-12.9_Test','h13-13.9_Test', 'i14-14.9_Test','j15-15.9_Test', 'k16-16.9_Test', 'l17-17.9_Test','m18-18.9_Test','n19-19.9_Test']


def extract_sliding_window_segments(image):
	"""
	Extract sliding window segments from an image's unmasked (tire tread) area.
	Returns list of segments with shape (IMG_SIZE, IMG_SIZE, channels).
	"""
	# Define the unmasked region bounds
	unmasked_y_start = HEIGHT_OFFSET + MASK_TOP_ROWS * SEGMENT_HEIGHT
	unmasked_y_end = image.shape[0]
	unmasked_x_start = WIDTH_OFFSET + MASK_LEFT_COLS * SEGMENT_WIDTH
	unmasked_x_end = image.shape[1] - MASK_RIGHT_COLS * SEGMENT_WIDTH

	# Crop to unmasked region (tire tread area only)
	unmasked_region = image[unmasked_y_start:unmasked_y_end,
	                        unmasked_x_start:unmasked_x_end, :]

	segments = []
	region_height = unmasked_region.shape[0]
	region_width = unmasked_region.shape[1]

	y = 0
	while y + SEGMENT_HEIGHT <= region_height:
		x = 0
		while x + SEGMENT_WIDTH <= region_width:
			segment = unmasked_region[y:y+SEGMENT_HEIGHT, x:x+SEGMENT_WIDTH, :]
			segments.append(segment)
			x += STEP_SIZE
		y += STEP_SIZE

	return segments


def preprocess_input_fn(img):
	"""Preprocess image input with explicit data type handling"""
	img = tf.cast(img, tf.float32)

	if GRAYSCALE_INPUT:
		img = tf.image.rgb_to_grayscale(img)
		img = tf.image.grayscale_to_rgb(img)

	return img


def setup_combined_dataset(data_location=''):
	"""
	Pre-segment images and save segments in individual folders.
	Each original image gets its own folder with all segments inside.
	"""
	print("Setting up combined dataset with pre-segmented images...")

	# Use data_location if provided, otherwise use current directory
	combined_dir = os.path.join(data_location, COMBINED_DIR) if data_location else COMBINED_DIR

	if os.path.exists(combined_dir):
		print(f"Removing existing {combined_dir}...")
		import shutil
		shutil.rmtree(combined_dir)

	os.makedirs(combined_dir)
	print(f"Created {combined_dir}")

	# Process all training and test folders
	for folder in TRAIN_FOLDERS + TEST_FOLDERS:
		folder_path = os.path.join(data_location, folder)
		if not os.path.exists(folder_path):
			print(f"Warning: Folder not found - {folder_path}")
			continue

		class_label = folder.split('_')[0]
		print(f"\nProcessing {folder} (class: {class_label})...")

		files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
		processed_count = 0
		total_segments = 0

		for img_file in files:
			src_path = os.path.join(folder_path, img_file)
			img_name = os.path.splitext(img_file)[0]

			# Create image-specific folder
			img_folder = os.path.join(combined_dir, f"{class_label}_{img_name}")
			os.makedirs(img_folder, exist_ok=True)

			try:
				# Load and preprocess image
				image = tf.keras.preprocessing.image.load_img(src_path)
				image_array = tf.keras.preprocessing.image.img_to_array(image)
				image_array = preprocess_input_fn(image_array)
				image_array = image_array.numpy() if isinstance(image_array, tf.Tensor) else image_array

				# Extract segments
				segments = extract_sliding_window_segments(image_array)

				# Save each segment
				for idx, segment in enumerate(segments):
					segment_path = os.path.join(img_folder, f"segment_{idx:03d}.png")
					tf.keras.preprocessing.image.save_img(segment_path, segment)

				total_segments += len(segments)
				processed_count += 1
			except Exception as e:
				print(f"  Error processing {img_file}: {str(e)}")
				continue

		print(f"  Processed {processed_count} images, saved {total_segments} segments")

	print("\n✓ Combined dataset setup complete!")


def split_data_kfold(data_location=''):
	"""
	Split the pre-segmented dataset into K folds.
	Returns list of folds where each fold contains image folder paths.
	"""
	print(f"\nSplitting dataset into {K_CROSS_FOLDS} K-folds...")

	# Use data_location if provided, otherwise use COMBINED_DIR
	combined_dir = os.path.join(data_location, 'combined_data_directory') if data_location else COMBINED_DIR
	combined_path = pathlib.Path(combined_dir)
	all_image_folders = sorted(list(combined_path.glob('*')))

	if not all_image_folders:
		raise ValueError(f"No image folders found in {combined_dir}")

	np.random.shuffle(all_image_folders)
	num_images = len(all_image_folders)
	fold_size = num_images // K_CROSS_FOLDS

	folds = []
	for i in range(K_CROSS_FOLDS):
		start_idx = i * fold_size
		if i == K_CROSS_FOLDS - 1:
			# Last fold gets remaining images
			fold = all_image_folders[start_idx:]
		else:
			fold = all_image_folders[start_idx:start_idx + fold_size]
		folds.append(fold)

	print(f"Total images: {num_images}")
	print(f"Fold size: {fold_size}")
	print(f"Total folds: {len(folds)}")
	for i, fold in enumerate(folds):
		print(f"  Fold {i}: {len(fold)} images")

	return folds


def load_segments_from_folder(folder_path):
	"""
	Load all segments from an image folder and stack them.
	Returns array of shape (num_segments, IMG_SIZE, IMG_SIZE, 3).
	"""
	segment_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

	segments = []
	for seg_file in segment_files:
		seg_path = os.path.join(folder_path, seg_file)
		# Use PIL directly — tf.keras.preprocessing.image wrappers accumulate
		# internal TF references when called inside tf.py_function workers,
		# causing gradual RAM growth that crashes the node around epoch 8.
		with Image.open(seg_path) as img:
			img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert('RGB')
			img_array = np.array(img, dtype=np.uint8)
		segments.append(img_array)

	if not segments:
		raise ValueError(f"No segments found in {folder_path}")

	# Pad or downsample to exactly SEGMENTS_PER_IMAGE segments
	# Store as uint8 (1 byte/pixel) instead of float32 (4 bytes/pixel) to keep
	# in-flight loader worker RAM at ~44 MB per image instead of ~177 MB.
	segments = np.array(segments, dtype=np.uint8)

	if len(segments) < SEGMENTS_PER_IMAGE:
		# Pad with zeros
		pad_size = SEGMENTS_PER_IMAGE - len(segments)
		padding = np.zeros((pad_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
		segments = np.concatenate([segments, padding], axis=0)
	elif len(segments) > SEGMENTS_PER_IMAGE:
		# Evenly sample across the full tread span instead of taking only early windows.
		indices = np.linspace(0, len(segments) - 1, SEGMENTS_PER_IMAGE, dtype=np.int32)
		segments = segments[indices]

	return segments


def create_dataset_from_folds(folds, fold_index, batch_size, augment=False, num_replicas=1,
                               num_parallel_calls=4, prefetch_size=2):
	"""
	Create train, validation, and test datasets from K-fold split.
	fold_index: which fold to use as test set (0 to K-1)
	Returns: (train_dataset, val_dataset, test_dataset)
	"""
	print(f"\nCreating datasets for K-fold iteration {fold_index}...")

	# Split into test fold and training folds
	test_folders = list(folds[fold_index])
	train_val_folders = []
	for i, fold in enumerate(folds):
		if i != fold_index:
			train_val_folders.extend(fold)

	# Further split training into train and validation (80/20)
	np.random.shuffle(train_val_folders)
	val_split = int(0.2 * len(train_val_folders))
	val_folders = train_val_folders[:val_split]
	train_folders = train_val_folders[val_split:]

	# Compute effective batch size (must be divisible by num_replicas)
	effective_batch_size = batch_size
	if num_replicas > 1 and (effective_batch_size % num_replicas != 0):
		adjusted = (effective_batch_size // num_replicas) * num_replicas
		if adjusted == 0:
			adjusted = num_replicas
		print(
			f"  [Batch Adjust] Requested batch_size={effective_batch_size} is not divisible "
			f"by num_replicas={num_replicas}. Using {adjusted}."
		)
		effective_batch_size = adjusted

	train_steps = max(1, len(train_folders) // effective_batch_size)
	val_steps = max(1, len(val_folders) // effective_batch_size)
	test_steps = max(1, len(test_folders) // effective_batch_size)

	print(f"  Training samples: {len(train_folders)} → {train_steps} steps/epoch (batch {effective_batch_size})")
	print(f"  Validation samples: {len(val_folders)} → {val_steps} steps/epoch (batch {effective_batch_size})")
	print(f"  Test samples: {len(test_folders)} → {test_steps} steps (batch {effective_batch_size})")

	def create_segment_dataset(folder_paths, is_training=False):
		"""Parallel tf.data pipeline – loads image folders concurrently to keep GPU fed."""
		folder_paths_str = [str(fp) for fp in folder_paths]

		label_list = []
		for folder_path in folder_paths:
			folder_name = os.path.basename(str(folder_path))
			class_label = folder_name.split('_')[0]
			class_idx = ord(class_label[0]) - ord('a')
			label_list.append(class_idx)

		dataset = tf.data.Dataset.from_tensor_slices(
			(folder_paths_str, np.array(label_list, dtype=np.int32))
		)

		if is_training:
			dataset = dataset.shuffle(
				buffer_size=len(folder_paths_str),
				reshuffle_each_iteration=True
			)

		def load_fn(folder_path_tensor, label):
			segments = tf.py_function(
				func=lambda p: load_segments_from_folder(p.numpy().decode('utf-8')),
				inp=[folder_path_tensor],
				Tout=tf.uint8  # uint8 = 44 MB/image vs float32 = 177 MB/image
			)
			segments.set_shape((SEGMENTS_PER_IMAGE, IMG_SIZE, IMG_SIZE, 3))
			return segments, label

		# Limit concurrent loaders to bound peak RAM usage based on detected GPU memory.
		# num_parallel_calls is passed in from the caller (derived from GPU capacity).
		dataset = dataset.map(load_fn, num_parallel_calls=num_parallel_calls)

		if is_training and augment:
			# Use pure tf.image ops instead of tf.keras.Sequential augmentation.
			# Keras stateful preprocessing layers called from AUTOTUNE-parallel map
			# workers can trigger concurrent build() calls, each creating new TF
			# variables/ops that accumulate in the default graph and leak RAM.
			def augment_fn(segments, label):
				segments = tf.cast(segments, tf.float32)
				segments = tf.image.random_flip_left_right(segments)
				segments = tf.image.random_flip_up_down(segments)
				# Brightness: ±10% of [0, 255] range
				delta = tf.random.uniform((), -25.5, 25.5)
				segments = tf.clip_by_value(segments + delta, 0.0, 255.0)
				return segments, label

			# Cap parallelism to prevent AUTOTUNE from spawning many threads,
			# each holding a 177 MB float32 tensor in flight simultaneously.
			dataset = dataset.map(augment_fn, num_parallel_calls=num_parallel_calls)

		# Scale [0, 255] → [-1, 1] as expected by MobileNetV3 ImageNet weights.
		# Cast to float32 here for the non-augmented (val/test) path where segments
		# are still uint8 coming out of load_fn.
		dataset = dataset.map(
			lambda x, y: (
				tf.keras.applications.mobilenet_v3.preprocess_input(tf.cast(x, tf.float32)),
				tf.one_hot(y, NUM_CLASSES)
			),
			num_parallel_calls=tf.data.AUTOTUNE
		)

		# Keep per-step shape consistent across replicas to avoid GPU fused-op shape bugs.
		dataset = dataset.batch(effective_batch_size, drop_remainder=True)
		# Limit prefetch: each batch is batch_size × 177 MB (float32 post-preprocess).
		# prefetch_size is passed in from the caller (derived from GPU capacity).
		dataset = dataset.prefetch(prefetch_size)

		return dataset

	train_dataset = create_segment_dataset(train_folders, is_training=True).repeat()
	val_dataset = create_segment_dataset(val_folders, is_training=False).repeat()
	test_dataset = create_segment_dataset(test_folders, is_training=False)

	return train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps


def build_model():
	"""
	Build a MobileNetV3 model that takes all 294 segments per image.
	Input shape: (SEGMENTS_PER_IMAGE, IMG_SIZE, IMG_SIZE, 3)
	Output: class predictions (NUM_CLASSES)
	"""
	# Create input for all segments
	segment_input = layers.Input(shape=(SEGMENTS_PER_IMAGE, IMG_SIZE, IMG_SIZE, 3),
	                             name='segment_input')

	# Load pre-trained MobileNetV3 for single image processing
	base_model = MobileNetV3Large(
		input_shape=(IMG_SIZE, IMG_SIZE, 3),
		include_top=False,
		weights='imagenet'
	)
	base_model.trainable = False

	# Apply backbone per segment with shared weights.
	x = layers.TimeDistributed(base_model)(segment_input)

	# Global average pooling across spatial dimensions for each segment
	# Shape: (batch_size, SEGMENTS_PER_IMAGE, base_model_features)
	x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

	# Aggregate segments: average across all segments
	# Shape: (batch_size, base_model_features)
	x = layers.GlobalAveragePooling1D()(x)

	# Classification head
	x = layers.BatchNormalization()(x)
	x = layers.Dense(256, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(128, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.3)(x)
	# Keep classifier output in float32 for numerical stability.
	outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

	model = models.Model(inputs=segment_input, outputs=outputs)

	return model


def fine_tune_model(model):
	"""Unfreeze some layers of the base model for fine-tuning"""
	# Find TimeDistributed layer and get the underlying base model
	for layer in model.layers:
		if isinstance(layer, layers.TimeDistributed) and hasattr(layer, 'layer'):
			base_model = layer.layer
			if 'mobilenetv3' in base_model.name.lower():
				# Unfreeze last 50 layers
				base_model.trainable = True
				for layer_base in base_model.layers[:-50]:
					layer_base.trainable = False
				break


def train_model(model, train_dataset, val_dataset, learning_rate, fold_index,
				batch_size_val, total_iterations, output_dir=None, strategy=None,
				steps_per_epoch=None, validation_steps=None):
	"""
	Train the model with the given hyperparameters.
	output_dir: directory to save model checkpoints (defaults to DATA_LOCATION)
	"""
	if output_dir is None:
		output_dir = DATA_LOCATION

	# Ensure output directory exists
	os.makedirs(output_dir, exist_ok=True)

	print(f"\n{'='*60}")
	print(f"Training - Fold: {fold_index+1}/{K_CROSS_FOLDS}, "
	      f"LR: {learning_rate}, Batch Size: {batch_size_val}")
	print(f"{'='*60}")

	# Create callbacks
	model_name = f"fold_{fold_index+1}_lr_{learning_rate}_bs_{batch_size_val}"

	class _EpochGCCallback(tf.keras.callbacks.Callback):
		"""Force Python garbage collection at each epoch end to prevent RAM
		   accumulation from TF/Python objects that outlive their useful lifetime."""
		def on_epoch_end(self, epoch, logs=None):
			gc.collect()

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath=os.path.join(output_dir, f'best_model_{model_name}.keras'),
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
			factor=0.5,
			patience=5,
			min_delta=0.0001,
			min_lr=learning_rate / 100
		),
		tf.keras.callbacks.TerminateOnNaN(),
		_EpochGCCallback(),
	]

	# Compile model (inside strategy scope when available).
	if strategy is not None:
		with strategy.scope():
			model.compile(
				optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
				loss='categorical_crossentropy',
				metrics=['accuracy'],
				jit_compile=False
			)
	else:
		model.compile(
			optimizer=optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
			loss='categorical_crossentropy',
			metrics=['accuracy'],
			jit_compile=False
		)

	# Train
	history = model.fit(
		train_dataset,
		epochs=EPOCHS,
		steps_per_epoch=steps_per_epoch,
		validation_data=val_dataset,
		validation_steps=validation_steps,
		callbacks=callbacks
	)

	return history


def evaluate_model(model, test_dataset, steps=None):
	"""Evaluate the model on test data"""
	print("\n" + "="*60)
	print("Evaluating on test set...")
	print("="*60)

	# Standard evaluation (no aggregation needed since we pass all segments directly)
	test_loss, test_accuracy = model.evaluate(test_dataset, steps=steps)

	print(f"\nTest Loss: {test_loss:.4f}")
	print(f"Test Accuracy: {test_accuracy:.4f}")

	return test_loss, test_accuracy


def plot_training_history(history, filename, output_dir=None):
	"""Plot training and validation metrics
	filename: just the filename (e.g., 'training_history.png')
	output_dir: optional directory to save (defaults to DATA_LOCATION if filename is just a name)
	"""
	# If filename contains path separators, use it as-is; otherwise combine with output_dir
	if os.path.dirname(filename):
		filepath = filename
	else:
		if output_dir is None:
			output_dir = DATA_LOCATION
		os.makedirs(output_dir, exist_ok=True)
		filepath = os.path.join(output_dir, filename)

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
	plt.savefig(filepath)
	plt.close()


def calculate_stepped_accuracy(y_true, y_pred):
	"""Calculate stepped accuracy metrics"""
	y_true_idx = np.argmax(y_true, axis=1)
	y_pred_idx = np.argmax(y_pred, axis=1)

	class_distance = np.abs(y_true_idx - y_pred_idx)

	exact = np.mean(class_distance == 0)
	within_1 = np.mean(class_distance <= 1)
	within_2 = np.mean(class_distance <= 2)
	within_3 = np.mean(class_distance <= 3)

	return exact, within_1, within_2, within_3


# ---------------------------------------------------------------------------
# Feature pre-extraction (optional but major speedup for frozen-backbone phase)
# ---------------------------------------------------------------------------
# Because the backbone is frozen during initial training, running 294 MobileNetV3
# forward passes per image every step is pure redundant compute.  Pre-extract
# once, save a single (SEGMENTS_PER_IMAGE, FEATURE_DIM) .npy per image folder,
# and train only the small classification head.  Fine-tuning still requires the
# full image pipeline (use build_model / create_dataset_from_folds for that).
# Usage:
#   preextract_and_save_features(DATA_LOCATION)
#   folds = split_data_kfold(DATA_LOCATION)
#   train_ds, val_ds, test_ds, ts, vs, ss = create_feature_dataset_from_folds(folds, fold_idx, batch_size)
#   model = build_feature_model()
# ---------------------------------------------------------------------------

def preextract_and_save_features(data_location):
	"""
	Run MobileNetV3Large on every segment folder and save a features.npy file.
	Each file has shape (SEGMENTS_PER_IMAGE, FEATURE_DIM).  Already-extracted
	folders are skipped, so re-running is safe.
	"""
	print("Pre-extracting MobileNetV3Large backbone features...")
	backbone = MobileNetV3Large(
		input_shape=(IMG_SIZE, IMG_SIZE, 3),
		include_top=False,
		weights='imagenet'
	)
	feature_extractor = tf.keras.Sequential([backbone, layers.GlobalAveragePooling2D()])
	feature_extractor.trainable = False

	combined_dir = os.path.join(data_location, 'combined_data_directory')
	image_dirs = sorted([
		d for d in os.listdir(combined_dir)
		if os.path.isdir(os.path.join(combined_dir, d))
	])
	total = len(image_dirs)
	print(f"  Found {total} image folders.")

	SUB_BATCH = 32  # segments per GPU sub-batch; lower if OOM
	for i, img_dir_name in enumerate(image_dirs):
		img_dir = os.path.join(combined_dir, img_dir_name)
		feature_file = os.path.join(img_dir, 'features.npy')

		if os.path.exists(feature_file):
			continue

		segments = load_segments_from_folder(img_dir)  # (SEGMENTS_PER_IMAGE, H, W, 3)
		segments_t = tf.keras.applications.mobilenet_v3.preprocess_input(
			tf.constant(segments, dtype=tf.float32)
		)

		all_features = []
		for start in range(0, SEGMENTS_PER_IMAGE, SUB_BATCH):
			end = min(start + SUB_BATCH, SEGMENTS_PER_IMAGE)
			feats = feature_extractor(segments_t[start:end], training=False)
			all_features.append(feats.numpy())

		np.save(feature_file, np.concatenate(all_features, axis=0))

		if (i + 1) % 50 == 0 or (i + 1) == total:
			print(f"  [{i+1}/{total}] features extracted")

	print("✓ Feature extraction complete.")


def load_features_from_folder(folder_path):
	"""Load pre-extracted backbone features from a .npy file."""
	feature_file = os.path.join(folder_path, 'features.npy')
	if not os.path.exists(feature_file):
		raise FileNotFoundError(
			f"No features.npy in {folder_path}. Run preextract_and_save_features() first."
		)
	return np.load(feature_file).astype(np.float32)


def build_feature_model():
	"""
	Classification head that takes pre-extracted features as input.
	Input shape: (SEGMENTS_PER_IMAGE, FEATURE_DIM)
	~300x less I/O and no backbone forward pass per training step.
	"""
	feature_input = layers.Input(shape=(SEGMENTS_PER_IMAGE, FEATURE_DIM), name='feature_input')
	x = layers.GlobalAveragePooling1D()(feature_input)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(256, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Dense(128, activation='relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.3)(x)
	outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
	return models.Model(inputs=feature_input, outputs=outputs)


def create_feature_dataset_from_folds(folds, fold_index, batch_size, num_replicas=1, prefetch_size=2):
	"""
	Like create_dataset_from_folds but loads pre-extracted .npy features.
	Must call preextract_and_save_features() first.
	Returns: (train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps)
	"""
	print(f"\nCreating feature datasets for K-fold iteration {fold_index}...")

	test_folders = list(folds[fold_index])
	train_val_folders = []
	for i, fold in enumerate(folds):
		if i != fold_index:
			train_val_folders.extend(fold)

	np.random.shuffle(train_val_folders)
	val_split = int(0.2 * len(train_val_folders))
	val_folders = train_val_folders[:val_split]
	train_folders = train_val_folders[val_split:]

	effective_batch_size = batch_size
	if num_replicas > 1 and (effective_batch_size % num_replicas != 0):
		adjusted = max(num_replicas, (effective_batch_size // num_replicas) * num_replicas)
		print(f"  [Batch Adjust] {effective_batch_size} → {adjusted} (divisible by {num_replicas})")
		effective_batch_size = adjusted

	train_steps = max(1, len(train_folders) // effective_batch_size)
	val_steps   = max(1, len(val_folders)   // effective_batch_size)
	test_steps  = max(1, len(test_folders)  // effective_batch_size)

	print(f"  Training: {len(train_folders)} samples → {train_steps} steps/epoch")
	print(f"  Validation: {len(val_folders)} samples → {val_steps} steps/epoch")
	print(f"  Test: {len(test_folders)} samples → {test_steps} steps")

	def create_feature_dataset(folder_paths, is_training=False):
		folder_paths_str = [str(fp) for fp in folder_paths]
		label_list = []
		for folder_path in folder_paths:
			folder_name = os.path.basename(str(folder_path))
			class_label = folder_name.split('_')[0]
			label_list.append(ord(class_label[0]) - ord('a'))

		dataset = tf.data.Dataset.from_tensor_slices(
			(folder_paths_str, np.array(label_list, dtype=np.int32))
		)

		if is_training:
			dataset = dataset.shuffle(buffer_size=len(folder_paths_str), reshuffle_each_iteration=True)

		def load_fn(folder_path_tensor, label):
			features = tf.py_function(
				func=lambda p: load_features_from_folder(p.numpy().decode('utf-8')),
				inp=[folder_path_tensor],
				Tout=tf.float32
			)
			features.set_shape((SEGMENTS_PER_IMAGE, FEATURE_DIM))
			return features, label

		dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
		dataset = dataset.map(
			lambda x, y: (x, tf.one_hot(y, NUM_CLASSES)),
			num_parallel_calls=tf.data.AUTOTUNE
		)
		dataset = dataset.batch(effective_batch_size, drop_remainder=True)
		# Cap prefetch to avoid buffering many large feature batches in GPU memory.
		dataset = dataset.prefetch(prefetch_size)
		return dataset

	train_dataset = create_feature_dataset(train_folders, is_training=True).repeat()
	val_dataset   = create_feature_dataset(val_folders,   is_training=False).repeat()
	test_dataset  = create_feature_dataset(test_folders,  is_training=False)

	return train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps