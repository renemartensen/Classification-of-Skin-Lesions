import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical

class Dataloader(Sequence):
    def __init__(self, data, batch_size, image_size, class_distribution, is_validation=False, preprocess_function=None):
        self.data = data  # List of (path, label) tuples
        if is_validation: 
            self.num_of_samples = len(data)
        else: 
            self.num_of_samples = sum(class_distribution)
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_distribution = class_distribution
        self.is_validation = is_validation
        self.preprocess_function = preprocess_function
        self.num_classes = len(self.class_distribution)  # Assuming valid distribution is provided.
        self.cumulative_sampled_counts = [0] * self.num_classes
        # Use augmentation only for training
        if not self.is_validation:
            self.classimages_per_batch = self._randomized_imgs_per_class_per_batch()
            self.datagen = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                shear_range=50, 
                height_shift_range=0.2,
                width_shift_range=0.2,
                rotation_range=360, 
                brightness_range=[0.3, 1.8], 
                channel_shift_range=random.uniform(20, 50),
                zoom_range=0.3
            )
        else:
            self.datagen = None  # No augmentation for validation and test
            self.classimages_per_batch = None
            
        self.indexes_by_class = [[] for _ in range(self.num_classes)]
        self._setup_class_indexes()

        print(f"Initialized dataloader with {self.num_of_samples} data samples and batch size {self.batch_size}.")
        print(f"Class distribution: {self.class_distribution}")
        print(f"Images per class per batch: {self.classimages_per_batch}")
        

    def _setup_class_indexes(self):
        """Set up a mapping from each class to its image indices."""
        class_counts = [0] * self.num_classes
        for idx, (_, label) in enumerate(self.data):
            self.indexes_by_class[label].append(idx)
            class_counts[label] += 1

    def _randomized_imgs_per_class_per_batch(self):
        """
        Dynamically calculate class counts for the next batch based on remaining samples needed 
        to meet the overall class distribution. Add randomness to enhance variability.
        """
        # Remaining samples needed for each class
        remaining_counts = [
            desired - sampled
            for desired, sampled in zip(self.class_distribution, self.cumulative_sampled_counts)
        ]
        total_remaining = sum(remaining_counts)

        if total_remaining <= 0:  # If all samples are used up, return zeros
            return [0] * self.num_classes

        # Dynamically adjust normalized distribution
        normalized_class_distribution = [
            remaining / total_remaining if total_remaining > 0 else 0
            for remaining in remaining_counts
        ]

        # Generate randomized counts for this batch
        scaled_class_counts = [
            int(class_ratio * self.batch_size) for class_ratio in normalized_class_distribution
        ]

        # Add randomness to the scaled counts
        for i in range(len(scaled_class_counts)):
            max_adjust = min(2, remaining_counts[i])  # Don't exceed remaining samples
            adjustment = np.random.randint(-max_adjust, max_adjust + 1)
            scaled_class_counts[i] = max(0, scaled_class_counts[i] + adjustment)

        # Adjust totals to match the batch size
        while sum(scaled_class_counts) > self.batch_size:
            for i in range(len(scaled_class_counts)):
                if scaled_class_counts[i] > 0:
                    scaled_class_counts[i] -= 1
                    if sum(scaled_class_counts) == self.batch_size:
                        break

        while sum(scaled_class_counts) < self.batch_size:
            for i in range(len(scaled_class_counts)):
                if remaining_counts[i] > 0:  # Only add to classes with remaining samples
                    scaled_class_counts[i] += 1
                    if sum(scaled_class_counts) == self.batch_size:
                        break

        return scaled_class_counts

    def __getitem__(self, index):
        """Sample balanced batches with randomized class counts."""
        start = index * self.batch_size
        if start + self.batch_size > self.num_of_samples:
            end = self.num_of_samples  # Cap at the last image
            batch_indexes = range(start, end)
        else:
            batch_indexes = range(start, start + self.batch_size)

        # For validation, sample sequentially
        if self.is_validation:
            batch_indexes = list(batch_indexes)
        else:
            batch_indexes =[]
            batch_class_counts = self._randomized_imgs_per_class_per_batch()
            for class_idx, num_samples in enumerate(batch_class_counts):
                available_samples = self.indexes_by_class[class_idx]
                if len(available_samples) > 0:
                    sampled = random.sample(available_samples, min(len(available_samples), num_samples))
                    batch_indexes.extend(sampled)
                    self.cumulative_sampled_counts[class_idx] += len(sampled)  # Update sampled counts

        batch_images, batch_labels = [], []

        for idx in batch_indexes:
            image = self._load_and_preprocess_image(self.data[idx][0])  # Load image path from data tuple
            batch_images.append(image)
            batch_labels.append(to_categorical(self.data[idx][1], num_classes=self.num_classes))

        return np.array(batch_images), np.array(batch_labels)

    def __len__(self):
        """Calculate the total number of batches."""
        return (sum(self.class_distribution) + self.batch_size - 1) // self.batch_size

    def _load_and_preprocess_image(self, image_path):
        assert os.path.exists(image_path), f"Image path does not exist: {image_path}"
        img = load_img(image_path, target_size=self.image_size)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        if self.preprocess_function:
            x = self.preprocess_function(x)
        
        if self.datagen:  # Apply augmentation if in training
            x = next(self.datagen.flow(x, batch_size=1))[0]
        
        return x



class DataloaderFactory():
    def __init__(self, dir, image_size, batch_size, set_distribution, class_distribution=[], preprocess_function=None):
        self.dir = dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.set_distribution = set_distribution
        self.class_distribution = class_distribution
        self.preprocess_function = preprocess_function

    def _load_data(self):
        """Load images and labels as tuples (path, label)."""
        data = []
        class_names = sorted(os.listdir(self.dir))
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(self.dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                data.append((image_path, label))  # Append (path, label) as a tuple
        return data

    def _split_data(self, data):
        """Split data into train, validation, and test sets based on the specified distribution."""
        indices = list(range(len(data)))
        _, val_ratio, test_ratio = [p / 100 for p in self.set_distribution]

        labels = [label for _, label in data]
        train_idxs, test_idxs, _, _ = train_test_split(
            indices, labels, test_size=(val_ratio + test_ratio), stratify=labels
        )
        val_split = val_ratio / (val_ratio + test_ratio)
        val_idxs, test_idxs = train_test_split(
            test_idxs, test_size=(1 - val_split), stratify=[labels[i] for i in test_idxs]
        )

        return {"train": train_idxs, "val": val_idxs, "test": test_idxs}

    def get_dataloaders(self):
        """Generate dataloaders for training, validation, and testing."""
        data = self._load_data()  # List of (path, label) tuples
        data_splits = self._split_data(data)  # Split data by indices
        loaders = {}

        for split, indexes in data_splits.items():
            preprocess_func = self.preprocess_function if split == "train" else None
            split_data = [data[i] for i in indexes]  # Subset the data using indices
            loaders[split] = Dataloader(
                data=split_data,  # Pass the list of tuples
                batch_size=self.batch_size,
                image_size=self.image_size,
                is_validation=(split != "train"),
                preprocess_function=preprocess_func,
                class_distribution=self.class_distribution,
            )

        return loaders["train"], loaders["val"], loaders["test"]