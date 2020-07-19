import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_to_zip = 'E:\Final_Year_Project'
PATH = os.path.join(os.path.dirname(path_to_zip), 'training_model')

train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'test')

train_AP2_dir = os.path.join(train_dir, 'AP2')  
train_AP3_dir = os.path.join(train_dir, 'AP3')
train_AP4_dir = os.path.join(train_dir, 'AP4')
train_AP5_dir = os.path.join(train_dir, 'AP5')
train_PLAX_dir = os.path.join(train_dir, 'PLAX')
train_PSAXAP_dir = os.path.join(train_dir, 'PSAX-AP')
train_PSAXAV_dir = os.path.join(train_dir, 'PSAX-AV')
train_PSAXMID_dir = os.path.join(train_dir, 'PSAX-MID')
train_PSAXMV_dir = os.path.join(train_dir, 'PSAX-MV')# directory with our training pictures

validation_AP2_dir = os.path.join(validation_dir, 'AP2')  
validation_AP3_dir = os.path.join(validation_dir, 'AP3')
validation_AP4_dir = os.path.join(validation_dir, 'AP4')
validation_AP5_dir = os.path.join(validation_dir, 'AP5')
validation_PLAX_dir = os.path.join(validation_dir, 'PLAX')
validation_PSAXAP_dir = os.path.join(validation_dir, 'PSAX-AP')
validation_PSAXAV_dir = os.path.join(validation_dir, 'PSAX-AV')
validation_PSAXMID_dir = os.path.join(validation_dir, 'PSAX-MID')
validation_PSAXMV_dir = os.path.join(validation_dir, 'PSAX-MV')# directory with our validation pictures

num_AP2_tr = len(os.listdir(train_AP2_dir))
num_AP3_tr = len(os.listdir(train_AP3_dir))
num_AP4_tr = len(os.listdir(train_AP4_dir))
num_AP5_tr = len(os.listdir(train_AP5_dir))
num_PLAX_tr = len(os.listdir(train_PLAX_dir))
num_PSAXAP_tr = len(os.listdir(train_PSAXAP_dir))
num_PSAXAV_tr = len(os.listdir(train_PSAXAV_dir))
num_PSAXMID_tr = len(os.listdir(train_PSAXMID_dir))
num_PSAXMV_tr = len(os.listdir(train_PSAXMV_dir))

num_AP2_val = len(os.listdir(validation_AP2_dir))
num_AP3_val = len(os.listdir(validation_AP3_dir))
num_AP4_val = len(os.listdir(validation_AP4_dir))
num_AP5_val = len(os.listdir(validation_AP5_dir))
num_PLAX_val = len(os.listdir(validation_PLAX_dir))
num_PSAXAP_val = len(os.listdir(validation_PSAXAP_dir))
num_PSAXAV_val = len(os.listdir(validation_PSAXAV_dir))
num_PSAXMID_val = len(os.listdir(validation_PSAXMID_dir))
num_PSAXMV_val = len(os.listdir(validation_PSAXMV_dir))

total_train = num_AP2_tr + num_AP3_tr + num_AP4_tr + num_AP5_tr + num_PLAX_tr + num_PSAXAP_tr+ num_PSAXAV_tr + num_PSAXMID_tr + num_PSAXMV_tr
total_val = num_AP2_val + num_AP3_val + num_AP4_val + num_AP5_val + num_PLAX_val + num_PSAXAP_val+ num_PSAXAV_val + num_PSAXMID_val + num_PSAXMV_val

print('total training images:', total_train)
print('total test images:', total_val)

train_image_generator = ImageDataGenerator(rescale=1./255,validation_split = 0.2) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

batch_size = 32
epochs = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_data_gen =train_image_generator.flow_from_directory(batch_size=batch_size,
                                                          directory=train_dir,
                                                          color_mode='rgb',
                                                          shuffle=True,
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              shuffle=False,
                                                              class_mode='categorical')


print('Training class: ',train_data_gen.class_indices.keys())
print('Validation class: ',val_data_gen.class_indices.keys())

sample_training_images, _ = next(train_data_gen)

import random

random.shuffle(sample_training_images)

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

base_model = tf.keras.applications.VGG16(input_shape=(IMG_HEIGHT, IMG_WIDTH ,3),
                                               include_top=False)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(9, activation = 'softmax')
])

optimizer_model = tf.keras.optimizers.Adam(learning_rate=0.001, name='Adam', decay=0.0001)
loss_model = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer_model, loss="categorical_crossentropy", metrics=['accuracy'])

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_freq='epoch',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train// batch_size ,
    epochs=epochs,
    callbacks = [model_checkpoint_callback],
    validation_data=val_data_gen,
    validation_steps=total_val// batch_size 
)

best_model = tf.keras.models.load_model('/tmp/checkpoint')
best_model.summary()

Y_pred = best_model.predict_generator(val_data_gen, total_val // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(val_data_gen.classes, y_pred))
print('Classification Report')
#target_names = ["AP2","AP3","AP4","AP5","PLAX","PSAX-AP","PSAX-AV","PSAX-MID","PSAX-MV"]
#target_names = ['AP3','AP5','PSAX-AV']
target_names = list(val_data_gen.class_indices.keys())
print(classification_report(val_data_gen.classes, y_pred, target_names=target_names))

#Visualize Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
results = best_model.predict(val_data_gen, total_val // batch_size)

# convert from class probabilities to actual class predictions
predicted_classes = np.argmax(results, axis=1)

# Names of predicted classes
class_names = list(val_data_gen.class_indices.keys())
#class_names = ['AP3','AP5','PSAX-AV']

# Generate the confusion matrix
cnf_matrix = confusion_matrix(val_data_gen.classes, y_pred)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

plt.show()