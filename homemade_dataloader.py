import os
import numpy as np
import random
import keras
from itertools import chain
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
import matplotlib.pyplot as plt

class HomemadeDataloader(Sequence):
    def __init__(self, data_dir, batch_size, image_size, isValidation=False, preprocess_function=None, class_distribuition=[]):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.isValidation = isValidation
        self.class_distribution = class_distribuition
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

        self.class_names = sorted(os.listdir(data_dir))
        self.num_classes = len(self.class_names)
        self.class_to_label = {class_name: i for i, class_name in enumerate(self.class_names)} # mapping {'class1': 0, 'class2': 1, ...})

        self.image_paths, self.labels = self._load_data()

        self.indexes_for_class = [[] for _ in range(self.num_classes)]

        self.all_image_indices = self._create_indices_distr()
        self._shuffle_indices()



    def _create_indices_distr(self):
        for index, label in enumerate(self.labels):
            self.indexes_for_class[label].append(index) # I assume its list 4 has all the indexes for label class_5

        if self.class_distribution:
            if len(self.class_distribution) != self.num_classes:
                raise ValueError("The len of class dist doesnt match number of classes")

            for i, c in enumerate(self.indexes_for_class):
                curr_class_len = self.class_distribution[i]

                curr_class_indices = self.indexes_for_class[i]
                l = len(curr_class_indices)
                # Calculate how many times to repeat indices
                multiplier = curr_class_len // l
                rest = curr_class_len % l

                # Create a copy of the original list to extend
                original_list = curr_class_indices.copy()
                self.indexes_for_class[i] = []

                # Extend the list based on the multiplier
                for p in range(multiplier):
                    self.indexes_for_class[i].extend(original_list)

                # Add the remaining items
                self.indexes_for_class[i].extend(original_list[:rest])

        res = list(chain.from_iterable(self.indexes_for_class)) # flattens the list-of-lists to one list
        return res


    def __len__(self):
        num_samples = len(self.all_image_indices) # no_images=100 batchsize=20, then 100/20=5 number of iterations to get through the whole dataset
        return num_samples // self.batch_size

    def _load_data(self):
        image_paths = []
        labels = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image_paths.append(image_path)
                labels.append(self.class_to_label[class_name])

        print(f"Found {len(image_paths)} images belonging to {self.num_classes} classes (dist says {sum(self.class_distribution)})")
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
        random.shuffle(self.all_image_indices)

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
        self._shuffle_indices()


