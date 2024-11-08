from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import Callback
import numpy
import tensorflow as tf


class HomemadeModel():
    def __init__(self):
        pass

    def load_model(self, number_of_samples):
    
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers for classification
        x = base_model.output
        x = Flatten()(x)
        preds = Dense(7, activation='softmax')(x)

        # Define the complete model
        model = Model(inputs=base_model.input, outputs=preds)

        # Define learning rate bounds and other parameters
        lower_bound = 4.4e-4
        upper_bound = 1e-3
        total_epochs = 100
        half_cycle_multiple = 6  # Set the multiple between 2 and 10
        batch_size = 32  # Define batch size here
        steps_per_epoch = int(number_of_samples / batch_size)  # Ensure train_generator is defined

        # Calculate half-cycle and full-cycle lengths in terms of batches
        half_cycle_length = steps_per_epoch * half_cycle_multiple  # Half-cycle length in batches
        full_cycle_length = 2 * half_cycle_length  # Full-cycle length in batches

        learning_rates = []

        # Custom callback for cyclical learning rate adjustment
        class BatchLearningRateScheduler(Callback):
            def __init__(self, lower_bound, upper_bound, full_cycle_length):
                super(BatchLearningRateScheduler, self).__init__()
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
                self.full_cycle_length = full_cycle_length
                self.batch_count = 0

            def on_batch_end(self, batch, logs=None):
                # Calculate position within the cycle (oscillates between 0 and 1)
                cycle_position = numpy.abs((self.batch_count % self.full_cycle_length) / self.full_cycle_length - 0.5) * 2
                
                # Calculate learning rate based on the cycle position
                lr = self.lower_bound + (self.upper_bound - self.lower_bound) * (1 - cycle_position)
                
                # Set the new learning rate
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                
                # Append to learning rates history for plotting
                learning_rates.append(lr)
                self.batch_count += 1

        # Instantiate and add the per-batch learning rate scheduler callback
        lr_scheduler = BatchLearningRateScheduler(lower_bound=lower_bound, upper_bound=upper_bound, full_cycle_length=full_cycle_length)

        # Compile the model
        model.compile(optimizer=SGD(learning_rate=lower_bound), loss='categorical_crossentropy', metrics=['accuracy'])

        return model, lr_scheduler