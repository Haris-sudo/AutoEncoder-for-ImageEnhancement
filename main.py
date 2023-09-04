from model import *
#from data import * 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import imageio 

#Defining a generator to yield pairs of images from two separate generators
def pair_generator(gen1, gen2):
    while True:
        yield (gen1.next(), gen2.next())

# Define PSNR (Peak Signal-to-Noise Ratio) as a metric. 
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

# Set parameters for the image data generator, including normalization and validation split
data_gen_args = dict(rescale=1./255, validation_split=0.2)
ground_truth_data_gen = ImageDataGenerator(**data_gen_args)
label_datagen = ImageDataGenerator(**data_gen_args)

batch_size = 16
target_size = (256,256)
# Configure training generator for ground truth images
train_ground_truth_gen = ground_truth_data_gen.flow_from_directory(
    'data/ground_truth',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42,
    subset='training'
)
# Configure training generator for label images
train_label_gen = label_datagen.flow_from_directory(
    'data/label',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42,
    subset='training'
)
# Configure validation generator for ground truth images
val_ground_truth_generator = ground_truth_data_gen.flow_from_directory(
    'data/ground_truth',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42,
    subset='validation'
)
# Configure validation generator for label images
val_label_generator = label_datagen.flow_from_directory(
    'data/label',
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=True,
    seed=42,
    subset='validation'
)

# Create training and validation paired generators
train_generator = pair_generator(train_ground_truth_gen, train_label_gen)
val_generator = pair_generator(val_ground_truth_generator, val_label_generator)

# Build Model
input_shape = (256, 256, 3)
model = build_autoencoder(input_shape=input_shape)
model.summary()

# Compile the model with Adam optimizer, mean squared error loss, and PSNR as a metric.
model.compile(optimizer= tf.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=[psnr])
# Configure checkpoint to save best model weights based on validation loss
checkpoint = ModelCheckpoint('model_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
# Train the model using the paired generators
model.fit(
    train_generator,
    steps_per_epoch=len(train_ground_truth_gen),
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(val_ground_truth_generator),
    callbacks=[checkpoint]
)

# Number of images to save
num_images_to_save = 10

# Get the degraded images and target images from the validation generator
val_degraded_images, val_target_images = next(val_generator)

# Get the model's predictions
predictions = model.predict(val_degraded_images)

# Save the degraded images, target images, and predicted images
for i in range(num_images_to_save):
    # Degraded image
    imageio.imsave(f'results/degraded_image_{i}.png', val_degraded_images[i])

    # Target image
    imageio.imsave(f'results/target_image_{i}.png', val_target_images[i])

    # Predicted image
    pred = predictions[i]
    pred = (pred*255).astype('uint8')
    imageio.imsave(f'results/predicted_image_{i}.png', pred)