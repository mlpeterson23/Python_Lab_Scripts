import os
import csv
import subprocess
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

from Thesis_CNN_Functions import (
	setup_combined_dataset,
	split_data_kfold,
	create_dataset_from_folds,
	build_model,
	fine_tune_model,
	train_model,
	evaluate_model,
	plot_training_history,
	preextract_and_save_features,
	create_feature_dataset_from_folds,
	build_feature_model,
	K_CROSS_FOLDS,
	NUM_CLASSES,
	SEGMENTS_PER_IMAGE
)

def get_gpu_memory_gb():
	"""Query total GPU memory in GB via nvidia-smi. Returns None if unavailable."""
	try:
		result = subprocess.run(
			['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
			capture_output=True, text=True, timeout=10
		)
		if result.returncode == 0:
			lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
			if lines:
				return int(lines[0]) / 1024.0
	except Exception:
		pass
	return None


def compute_pipeline_params(gpu_memory_gb):
	"""
	Return (num_parallel_calls, prefetch_size) scaled to available GPU memory.
	Each image batch (batch_size=4, 294 segments) occupies ~710 MB float32 post-preprocess.
	Conservative defaults prevent the data pipeline from pinning multi-GB of GPU memory.
	"""
	if gpu_memory_gb is None or gpu_memory_gb < 8:
		return 2, 1
	elif gpu_memory_gb < 16:
		return 2, 2
	elif gpu_memory_gb < 32:
		return 4, 2
	else:
		return 4, 4


# Configuration
BATCH_SIZES = [4]  # Batch sizes to test
LEARNING_RATES = [0.001]  # Learning rates to test
EPOCHS = 100
DATA_SETUP = False
USE_MIXED_PRECISION = True
ENABLE_XLA = False
# Set to True to pre-extract backbone features once and train only the classification head.
# ~300x less I/O per step; no backbone forward pass during frozen training.
# After feature-model training, switch back to build_model() for fine-tuning.
USE_FEATURE_PREEXTRACTION = False
# Results tracking
DATA_LOCATION = os.path.join(os.environ["SCRATCH"], "LCC_UP") + "/"  # Set to your data location if using external path
RESULTS_FILE = os.path.join(DATA_LOCATION, 'training_results.csv')


def initialize_results_file():
	"""Create CSV file for tracking results"""
	if not os.path.exists(RESULTS_FILE):
		with open(RESULTS_FILE, 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow([
				'Timestamp', 'Fold', 'Learning_Rate', 'Batch_Size',
				'Train_Loss', 'Val_Loss', 'Train_Accuracy', 'Val_Accuracy',
				'Test_Loss', 'Test_Accuracy'
			])


def save_results(fold_idx, lr, batch_size, history, test_loss, test_accuracy):
	"""Save training results to CSV"""
	timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	train_loss = history.history['loss'][-1]
	val_loss = history.history['val_loss'][-1]
	train_acc = history.history['accuracy'][-1]
	val_acc = history.history['val_accuracy'][-1]

	with open(RESULTS_FILE, 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([
			timestamp, fold_idx+1, lr, batch_size,
			f"{train_loss:.6f}", f"{val_loss:.6f}",
			f"{train_acc:.6f}", f"{val_acc:.6f}",
			f"{test_loss:.6f}", f"{test_accuracy:.6f}"
		])


def main():
	"""Main training loop with K-fold cross-validation and hyperparameter tuning"""
	print("="*70)
	print("THESIS CNN - K-FOLD CROSS VALIDATION WITH HYPERPARAMETER TUNING")
	print("="*70)

	# GPU Configuration - DO THIS FIRST
	print("\nConfiguring GPU...")
	gpus = tf.config.list_physical_devices('GPU')
	print(f"Detected GPUs: {gpus}")

	if gpus:
		try:
			# Enable memory growth to avoid OOM errors
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, False)
			print(f"✓ Memory growth disabled for {len(gpus)} GPU(s)")

			# Distribute strategy for multiple GPUs
			if len(gpus) > 1:
				strategy = tf.distribute.MirroredStrategy()
				print(f"✓ Using MirroredStrategy for {len(gpus)} GPUs")
			else:
				strategy = None
			num_replicas = len(gpus)
		except RuntimeError as e:
			print(f"Could not set memory growth: {e}")
			strategy = None
			num_replicas = 1
	else:
		print("No GPUs detected - using CPU")
		strategy = None
		num_replicas = 1

	# Mixed precision can trigger unstable fused kernels in some TF/CUDA combinations.
	if USE_MIXED_PRECISION:
		try:
			policy = tf.keras.mixed_precision.Policy('mixed_float16')
			tf.keras.mixed_precision.set_global_policy(policy)
			print("✓ Enabled mixed precision training (float16)")
		except Exception as e:
			print(f"Could not enable mixed precision training: {e}")
	else:
		tf.keras.mixed_precision.set_global_policy('float32')
		print("✓ Using float32 precision (stability mode)")

	# TensorFlow graph optimization
	print("Configuring TensorFlow graph optimizations...")
	tf.config.run_functions_eagerly(False)  # Keep graph mode for speed
	tf.config.optimizer.set_jit(ENABLE_XLA)
	tf.config.optimizer.set_experimental_options({
		"layout_optimizer": True,
		"function_optimization": True,
		"arithmetic_optimization": True,
	})
	if ENABLE_XLA:
		print("✓ XLA JIT enabled")
	else:
		print("✓ XLA JIT disabled (stability mode)")

	# Query GPU memory and derive safe data-pipeline settings.
	# This prevents the prefetch / parallel-loader buffers from pinning
	# multiple GB of GPU memory and causing OOM before training even starts.
	gpu_memory_gb = get_gpu_memory_gb()
	if gpu_memory_gb is not None:
		print(f"✓ Detected GPU memory: {gpu_memory_gb:.1f} GB")
	else:
		print("  (Could not detect GPU memory via nvidia-smi; using conservative defaults)")
	pipeline_parallel_calls, pipeline_prefetch = compute_pipeline_params(gpu_memory_gb)
	print(f"  Pipeline: num_parallel_calls={pipeline_parallel_calls}, prefetch={pipeline_prefetch}")

	# Ensure DATA_LOCATION directory exists
	os.makedirs(DATA_LOCATION, exist_ok=True)

	# Initialize results tracking
	initialize_results_file()
	print(f"✓ Results will be saved to {RESULTS_FILE}")

	# Setup dataset - pre-segment images and save in folders
	if DATA_SETUP or not os.path.exists(os.path.join(DATA_LOCATION, 'combined_data_directory')):
		print("\n" + "="*70)
		print("SETTING UP DATASET")
		print("="*70)
		setup_combined_dataset(DATA_LOCATION)

	# Optional: pre-extract backbone features once to skip 294 MobileNetV3 forward
	# passes per image per training step (only valid while backbone is frozen).
	if USE_FEATURE_PREEXTRACTION:
		print("\n" + "="*70)
		print("PRE-EXTRACTING BACKBONE FEATURES")
		print("="*70)
		preextract_and_save_features(DATA_LOCATION)

	# Split data into K folds
	print("\n" + "="*70)
	print("SPLITTING DATA INTO K-FOLDS")
	print("="*70)
	folds = split_data_kfold(DATA_LOCATION)

	# Total iterations for display
	total_iterations = len(BATCH_SIZES) * len(LEARNING_RATES) * K_CROSS_FOLDS
	current_iteration = 0

	# K-fold cross-validation loop
	fold_accuracies = []
	for fold_idx in range(K_CROSS_FOLDS):
		print("\n" + "="*70)
		print(f"K-FOLD ITERATION {fold_idx + 1}/{K_CROSS_FOLDS}")
		print("="*70)

		fold_results = []

		# Hyperparameter tuning loop
		for batch_size in BATCH_SIZES:
			for learning_rate in LEARNING_RATES:
				current_iteration += 1
				effective_batch = batch_size * SEGMENTS_PER_IMAGE
				print(f"\n[{current_iteration}/{total_iterations}] "
					  f"Fold {fold_idx+1}, Batch Size: {batch_size}, LR: {learning_rate}")
				print(f"Effective segment batch (batch_size x {SEGMENTS_PER_IMAGE}): {effective_batch}")

				try:
					# Create datasets for this fold and hyperparameter combination
					if USE_FEATURE_PREEXTRACTION:
						train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps = create_feature_dataset_from_folds(
							folds, fold_idx, batch_size, num_replicas=num_replicas,
							prefetch_size=pipeline_prefetch
						)
					else:
						train_dataset, val_dataset, test_dataset, train_steps, val_steps, test_steps = create_dataset_from_folds(
							folds, fold_idx, batch_size, augment=True, num_replicas=num_replicas,
							num_parallel_calls=pipeline_parallel_calls, prefetch_size=pipeline_prefetch
						)
					print(f"  Steps — train: {train_steps}, val: {val_steps}, test: {test_steps}")

					# Build model
					if strategy is not None:
						with strategy.scope():
							model = build_feature_model() if USE_FEATURE_PREEXTRACTION else build_model()
					else:
						model = build_feature_model() if USE_FEATURE_PREEXTRACTION else build_model()

					# Train model
					print(f"\nTraining model...")
					history = train_model(
						model, train_dataset, val_dataset,
						learning_rate, fold_idx, batch_size, total_iterations,
						output_dir=DATA_LOCATION,
						strategy=strategy,
						steps_per_epoch=train_steps,
						validation_steps=val_steps
					)

					# Plot training history
					plot_filename = os.path.join(DATA_LOCATION, f'training_history_fold{fold_idx+1}_lr{learning_rate}_bs{batch_size}.png')
					plot_training_history(history, plot_filename)
					print(f"✓ Training history saved to {plot_filename}")

					# Fine-tune the model
					print(f"\nFine-tuning model...")
					fine_tune_model(model)
					if strategy is not None:
						# Recompile under strategy
						with strategy.scope():
							model.compile(
								optimizer=optimizers.Adam(learning_rate=learning_rate/20),
								loss='categorical_crossentropy',
								metrics=['accuracy'],
								jit_compile=False
							)
					else:
						model.compile(
							optimizer=optimizers.Adam(learning_rate=learning_rate/20),
							loss='categorical_crossentropy',
							metrics=['accuracy'],
							jit_compile=False
						)

					# Fine-tune training (10-20 epochs)
					history_finetune = model.fit(
						train_dataset,
						epochs=20,
						steps_per_epoch=train_steps,
						validation_data=val_dataset,
						validation_steps=val_steps,
						callbacks=[
							tf.keras.callbacks.EarlyStopping(
								monitor='val_loss',
								patience=5,
								restore_best_weights=True
							)
						]
					)

					# Evaluate on test set
					print(f"\nEvaluating on test set...")
					test_loss, test_accuracy = evaluate_model(model, test_dataset, steps=test_steps)

					# Save results
					save_results(fold_idx, learning_rate, batch_size, history, test_loss, test_accuracy)
					fold_results.append({
						'batch_size': batch_size,
						'lr': learning_rate,
						'test_acc': test_accuracy
					})

					# Save model
					model_name = os.path.join(DATA_LOCATION, f'model_fold{fold_idx+1}_lr{learning_rate}_bs{batch_size}.keras')
					model.save(model_name)
					print(f"✓ Model saved as {model_name}")

				except Exception as e:
					print(f"❌ Error during training: {str(e)}")
					import traceback
					traceback.print_exc()
					continue

		# Find best hyperparameters for this fold
		if fold_results:
			best_result = max(fold_results, key=lambda x: x['test_acc'])
			fold_accuracies.append(best_result['test_acc'])
			print(f"\nBest hyperparameters for Fold {fold_idx+1}:")
			print(f"  Batch Size: {best_result['batch_size']}")
			print(f"  Learning Rate: {best_result['lr']}")
			print(f"  Test Accuracy: {best_result['test_acc']:.4f}")

	# Print final summary
	print("\n" + "="*70)
	print("FINAL RESULTS")
	print("="*70)
	if fold_accuracies:
		print(f"Mean Test Accuracy across all folds: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
		print(f"Best fold accuracy: {np.max(fold_accuracies):.4f}")
		print(f"Worst fold accuracy: {np.min(fold_accuracies):.4f}")
	print(f"\n✓ Complete! Results saved to {RESULTS_FILE}")
	print("="*70)


if __name__ == '__main__':
	main()
