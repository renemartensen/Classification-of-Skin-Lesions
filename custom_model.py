from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class CustomModel(tf.keras.Model):
    def __init__(self, number_of_samples):
        super(CustomModel, self).__init__()
        
        # Load and configure the base model
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom layers for classification
        x = base_model.output
        x = Flatten()(x)
        preds = Dense(7, activation='softmax')(x)

        # Define the complete model
        self.model = Model(inputs=base_model.input, outputs=preds)

        # Set cyclical learning rate parameters
        self.lower_bound = 1e-4
        self.upper_bound = 1e-3
        self.half_cycle_multiple = 6
        self.batch_size = 32
        self.steps_per_epoch = int(number_of_samples / self.batch_size)
        self.half_cycle_length = self.steps_per_epoch * self.half_cycle_multiple
        self.full_cycle_length = 2 * self.half_cycle_length
        self.learning_rates = []
        self.class_weight = None
        self.norms = []
        self.epochs = 5

    def compile(self):
        # Compile the internal model
        self.model.compile(optimizer=SGD(learning_rate=self.lower_bound), loss='categorical_crossentropy', metrics=['accuracy'])

    def evaluate(self, data):
        # Evaluate the model on the validation set
        return self.model.evaluate(data)
    
    def predict(self, data):
        return self.model.predict(data)
    


    def fit_epochs(self, train_generator, validation_generator, epochs, checkpoint_path, lr=None, class_weight=None):

        self.epochs = epochs

        if lr is not None:
            self.lower_bound = lr[0]
            self.upper_bound = lr[1]
        
        if class_weight is not None:
            self.class_weight = class_weight

        class GradientLogger(Callback):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.gradient_norms_log = []
                self.norms = []

            def on_epoch_end(self, epoch, logs=None):
                with tf.GradientTape() as tape:
                    norm = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in self.model.trainable_variables]))
                    self.norms.append(norm.numpy())

            def get_norms(self):
                return self.norms

        gradient_logger = GradientLogger(self.model)

        

        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        early_stopping_callback = EarlyStopping(
            monitor='val_loss',  
            patience=40,          
            min_delta=0.001,
            start_from_epoch=5
        )

        # Train the model using the cyclical learning rate scheduler and checkpoint callback
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=1,
            class_weight=self.class_weight,
            callbacks=[self.lr_scheduler, checkpoint_callback, early_stopping_callback, gradient_logger]
        )

        self.norms = gradient_logger.get_norms()

        return history
    
    def plot_trainable_weights(self, norms, epochs):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, epochs + 1), norms, marker='o', label='Norm of Trainable Variables')
        plt.xlabel('Epoch')
        plt.ylabel('Norm')
        plt.title('Norm of Trainable Variables vs Epoch')
        plt.grid()
        plt.legend()
        plt.show()
    
    def evaluate(self, data):
        # Evaluate the model on the validation set
        return self.model.evaluate(data)

    def unfreeze(self):
        # Unfreeze all layers in the base model
        for layer in self.model.layers:
            layer.trainable = True
        print("All layers have been unfrozen.")

    def lr_find(self, train_generator, validation_generator, min_lr=1e-10, max_lr=0.01, epochs=10):
        steps_per_epoch = train_generator.samples / self.batch_size
        total_batches = steps_per_epoch * epochs
        
        # Custom callback for linear learning rate increase
        class LinearLRScheduler(Callback):
            def __init__(self, min_lr, max_lr, total_batches):
                super().__init__()
                self.min_lr = min_lr
                self.max_lr = max_lr
                self.total_batches = total_batches
                self.batch_count = 0
                self.lr_history = []
                self.loss_history = []

            def on_batch_end(self, batch, logs=None):
                # Log the loss and learning rate
                loss = logs.get('loss')
                self.loss_history.append(loss)
                
                # Incrementally increase learning rate
                self.batch_count += 1
                lr = self.min_lr + (self.max_lr - self.min_lr) * (self.batch_count / self.total_batches)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                self.lr_history.append(lr)

        # Instantiate the linear learning rate scheduler callback
        lr_scheduler_callback = LinearLRScheduler(min_lr, max_lr, total_batches)
        
        # Compile the model with the minimum learning rate
        self.model.compile(optimizer=SGD(learning_rate=min_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model with the learning rate scheduler to find optimal lr
        self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            batch_size=self.batch_size,
            class_weight=self.class_weight,
            callbacks=[lr_scheduler_callback],
            verbose=1
        )

        # Plot learning rate vs. loss
        plt.plot(lr_scheduler_callback.lr_history, lr_scheduler_callback.loss_history)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Loss vs. Learning Rate')
        plt.show()

    @property
    def lr_scheduler(self):
        # Custom cyclical learning rate scheduler
        class BatchLearningRateScheduler(tf.keras.callbacks.Callback):
            def __init__(self, lower_bound, upper_bound, full_cycle_length, learning_rates):
                super().__init__()
                self.lower_bound = lower_bound
                self.upper_bound = upper_bound
                self.full_cycle_length = full_cycle_length
                self.batch_count = 0
                self.learning_rates = learning_rates

            def on_batch_end(self, batch, logs=None):
                cycle_position = np.abs((self.batch_count % self.full_cycle_length) / self.full_cycle_length - 0.5) * 2
                lr = self.lower_bound + (self.upper_bound - self.lower_bound) * (1 - cycle_position)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                self.learning_rates.append(lr)
                self.batch_count += 1

        return BatchLearningRateScheduler(self.lower_bound, self.upper_bound, self.full_cycle_length, self.learning_rates)

    


# Example usage:
# model = CustomModel(number_of_samples=1000)
# model.compile()
# model.fit_epochs(epochs=10, lr=0.001)
# model.unfreeze()
# lr_history = model.lr_find()
