# data_gen_args = dict(validation_split=0.2)
# ground_truth_data_gen = ImageDataGenerator(**data_gen_args)
# label_datagen = ImageDataGenerator(**data_gen_args)
#
# batch_size = 16
# target_size = (500,500)
#
# train_ground_truth_gen = ground_truth_data_gen.flow_from_directory(
#     'data/ground_truth',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=True,
#     seed=42,
#     subset='training'
# ) 
#
# train_label_gen =label_datagen.flow_from_directory(
#     'data/label',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=True,
#     seed=42,
#     subset='training'
# )
#
# val_ground_truth_generator = ground_truth_data_gen.flow_from_directory(
#     'data/ground_truth',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=True,
#     seed=42,
#     subset='validation')
#
# val_label_generator = label_datagen.flow_from_directory(
#     'data/label',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode='categorical',
#     shuffle=True,
#     seed=42,
#     subset='validation')
#
# train_generator = zip(train_ground_truth_gen, train_label_gen)
# val_generator = zip(val_ground_truth_generator, val_label_generator)
#
# data_gen_args = dict(validation_split=0.2)
# ground_truth_data_gen = ImageDataGenerator(**data_gen_args)
#
# batch_size = 16
# target_size = (500,500)
#
# train_ground_truth_gen = ground_truth_data_gen.flow_from_directory(
#     'data/ground_truth',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=True,
#     seed=42,
#     subset='training'
# ) 
#
# val_ground_truth_generator = ground_truth_data_gen.flow_from_directory(
#     'data/ground_truth',
#     target_size=target_size,
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=True,
#     seed=42,
#     subset='validation')
#
# train_generator = train_ground_truth_gen
# val_generator = val_ground_truth_generator
#
# print(len(train_ground_truth_gen))
# print(len(val_ground_truth_generator))
#
# input_shape = (500, 500, 3)
# model = build_autoencoder(input_shape=input_shape)
# model.compile(optimizer= tf.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['psnr'], run_eagerly=True)
# model.fit(
#     train_generator,
#     steps_per_epoch=len(train_ground_truth_gen),
#     epochs=300,
#     validation_data=val_generator,
#     validation_steps=len(val_ground_truth_generator))
#
# import imageio
# # Number of images you want to save
# num_images_to_save = 10
#
# # Get the ground truth images and labels from the validation generator
# val_ground_truth_images, val_labels = next(val_ground_truth_generator)
#
# # Get the model's predictions
# predictions = model.predict(val_ground_truth_images)
#
# # Convert the predicted labels to their original format
# predicted_labels = np.argmax(predictions, axis=1)
#
# # Save the original images, ground truth labels, and predicted labels
# for i in range(num_images_to_save):
#     # Original image
#     imageio.imsave(f'results/original_image_{i}.png', val_ground_truth_images[i])
#
#     # Ground truth label
#     imageio.imsave(f'results/ground_truth_label_{i}.png', val_labels[i])
#
#     # Predicted label
#     imageio.imsave(f'results/predicted_label_{i}.png', predicted_labels[i])
