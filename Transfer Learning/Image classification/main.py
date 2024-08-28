from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
import os
import pandas as pd
import h5py, PIL
print(h5py.__version__)

os.chdir(os.path.dirname(__file__))

# # Load the CSV file containing image paths and labels
# data = pd.read_csv('challenges.csv')

# # Swap columns to match what the code expects
# data.columns = ['class', 'image']

# # Split the data into training, validation, and test sets
# train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='train/data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    directory='validation/data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    directory='test/data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False)

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the pre-trained base
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(4, activation='softmax')(x)  # Output layer for classification

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Unfreeze the last few layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

model.save('fine_tuned_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')