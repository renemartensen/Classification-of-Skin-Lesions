import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from itertools import chain

class Dataloader(Sequence):
    def __init__(self, data, batch_size, image_size, is_validation=False, preprocess_function=None, class_distribuition=[], class_names=[]):
        #self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.isValidation = is_validation
        self.class_distribution = class_distribuition if not is_validation else []
        self.preprocess_function = preprocess_function

        self.datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            shear_range=50, 
            height_shift_range=0.2,
            width_shift_range=0.2,
            rotation_range=360, 
            brightness_range=[0.3, 1.8], 
            channel_shift_range= random.uniform(20, 50),
            zoom_range=0.3)

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.class_to_label = {class_name: i for i, class_name in enumerate(self.class_names)} # mapping {'class1': 0, 'class2': 1, ...})

        self.image_paths, self.labels = zip(*data)
        self.image_paths = list(self.image_paths)  # Convert from tuple to list
        self.labels = list(self.labels)           # Convert from tuple to list

        self.indexes_for_class = [[] for _ in range(self.num_classes)]

        self.all_image_indices = self._create_indices_distr() if not self.isValidation else list(range(len(self.image_paths)))
        self.samples = len(self.all_image_indices)
        #self._shuffle_indices()
        print(f"Found {len(self.image_paths)} images belonging to {self.num_classes} classes (dist says {sum(self.class_distribution)})")


    def _create_indices_distr(self):
        """Create indices for each class and distribute them equally across batches."""
        for index, label in enumerate(self.labels):
            self.indexes_for_class[label].append(index)

        if self.class_distribution:
            if len(self.class_distribution) != self.num_classes:
                raise ValueError("The length of class distribution doesn't match the number of classes.")

            for i, class_indices in enumerate(self.indexes_for_class):
                required_count = self.class_distribution[i]
                current_count = len(class_indices)

                if current_count < required_count:
                    repeats = (required_count // current_count) + 1
                    self.indexes_for_class[i] = (class_indices * repeats)[:required_count]
                else:
                    self.indexes_for_class[i] = random.sample(class_indices, required_count)

        if self.isValidation:
            print("Validation set")
            return list(chain.from_iterable(self.indexes_for_class))
        print("Training set")
        per_class_batch_size = self.batch_size // self.num_classes
        per_class_batches = []
        remainder = self.batch_size % self.num_classes
        remainder_window = list(range(remainder))  # Initial window of classes for extra samples
        total_number_of_batches = sum(self.class_distribution) // self.batch_size
        
        for i, class_indices in enumerate(self.indexes_for_class):
            random.shuffle(class_indices)  # Shuffle within the class
            class_batches = []
            j = 0

            # Distribute batches for the current class
            p = 0
            batch_index = 0  # Tracks which batch we are creating for this class
            while j + per_class_batch_size <= len(class_indices):
                if batch_index >= total_number_of_batches:
                    break
                if batch_index % self.num_classes in remainder_window:
                    # This batch for the current class gets an extra sample
                    batch = class_indices[j:j + per_class_batch_size + 1]
                    p += 1
                else:
                    # Standard batch size
                    batch = class_indices[j:j + per_class_batch_size]
                
                class_batches.append(batch)
                j += per_class_batch_size
                batch_index += 1  # Move to the next batch
                # Update the remainder window to "slide" for the next class
            remainder_window = [(x + 1) % self.num_classes for x in remainder_window]
            per_class_batches.append(class_batches)


        # Combine batches
        combined_batches = list(chain.from_iterable(zip(*per_class_batches)))
        res = list(chain.from_iterable(combined_batches))  # Flatten the list
        return res
    # Everything does as i think it should do. The problem is thatwe produce a list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] from the abovee function 
    # and then if we have a batch size of 3 we get [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9] [0, 1, 2]] which is not what we want




    

    def _load_data(self):
        image_paths = []
        labels = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image_paths.append(image_path)
                labels.append(self.class_to_label[class_name])

        #print(f"Found {len(image_paths)} images belonging to {self.num_classes} classes (dist says {sum(self.class_distribution)})")
        return image_paths, labels

    def _preprocess_image(self, image_path):
      img = load_img(image_path, target_size=self.image_size)
      x = img_to_array(img)

      if self.preprocess_function:
        x = self.preprocess_function(x)

      if not self.isValidation:
        x = x.reshape((1,) + x.shape)
        x = next(self.datagen.flow(x, batch_size=1))[0]

      return x

    def _shuffle_indices(self):
        self.all_image_indices = self._create_indices_distr()

    def __len__(self):
        num_samples = len(self.all_image_indices) # no_images=100 batchsize=20, then 100/20=5 number of iterations to get through the whole dataset
        return num_samples // self.batch_size
    
    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        images = []
        labels = []

        batch_image_indices = self.all_image_indices[start:end]

        for image_index in batch_image_indices:
            image_path = self.image_paths[image_index]
            image = self._preprocess_image(image_path)
            images.append(image)
            labels.append(to_categorical(self.labels[image_index], num_classes=self.num_classes))

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if not self.isValidation:
            self._shuffle_indices()




class DataloaderFactory():
    def __init__(self, dir, image_size, batch_size, set_distribution, class_distribution=[], preprocess_function=None):
        self.dir = dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.set_distribution = set_distribution
        self.class_distribution = class_distribution
        self.preprocess_function = preprocess_function
        self.class_names = []

    def _load_data(self):
        """Load images and labels as tuples (path, label)."""
        data = []
        self.class_names = sorted(os.listdir(self.dir))
        for label, class_name in enumerate(self.class_names):
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
                class_distribuition=self.class_distribution,
                class_names=self.class_names
            )

        return loaders["train"], loaders["val"], loaders["test"]