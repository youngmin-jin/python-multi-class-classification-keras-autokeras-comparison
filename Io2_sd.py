import numpy as np
import tensorflow_datasets as tfds
import autokeras as ak
import tensorflow as tf
import keras_tuner
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from input_output import *
from tensorflow.keras.optimizers import Adam

# ------------- var ---------------
image_size = (224, 224)


# ------------- data ---------------
# Load the Stanford Dogs dataset
builder = tfds.builder("stanford_dogs")
builder.download_and_prepare()

ds_train = builder.as_dataset(split='train')
ds_test = builder.as_dataset(split='test')

num_classes = builder.info.features['label'].num_classes


# unpacked and resize
def img_label(example):
  image = example['image']
  label = example['label']
  return image, label

ds_train = ds_train.map(img_label)
ds_test = ds_test.map(img_label)

# preproces (resize, normalize, and one-hot encoded)
def preprocessing(image, label):
  image = tf.image.resize(image, [224, 224])
  image = image / 255.0
  label = tf.one_hot(label, num_classes)
  return image, label

ds_train = ds_train.map(preprocessing)
ds_test = ds_test.map(preprocessing)

# batch and optimize
batch_size = 32
ds_train = ds_train.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# ds_train = ds_train.cache().prefetch(tf.data.AUTOTUNE)
# ds_test = ds_test.cache().prefetch(tf.data.AUTOTUNE)

# ------------- model ---------------
# initialize a model
class Io2_create_model(keras_tuner.HyperModel):
  def build(self, hp):  
    # input layer
    inputs = tf.keras.Input(shape=image_size+(3,)) 

    # hidden layers
    x = tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", fill_value=0.0, interpolation="bilinear", seed=None)(inputs)
    x = tf.keras.layers.RandomFlip('horizontal')(x)
    x = tf.keras.layers.RandomRotation(0.1, fill_mode="reflect", fill_value=0.0, interpolation="bilinear", seed=None)(x)
    x = tf.keras.applications.EfficientNetB7(
        input_shape=(224,224,3)
        , include_top=False
        , weights='imagenet'
        , drop_connect_rate=hp.Choice("drop_connect_rate", values=[0.2, 0.5])
        , pooling='avg'
    )(x) 
  
    # output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  
  
    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hp.Choice("learning_rate", values=[3e-4, 2e-05, 1e-5])), metrics=['accuracy'])
    return model
    
  def fit(self, hp, model, *args, **kwargs):
    return model.fit(*args, **kwargs)

# apply grid search
Io2_model = keras_tuner.GridSearch(
  Io2_create_model()
  , objective='accuracy'
  , overwrite=True
)

# early stopping
es = tf.keras.callbacks.EarlyStopping(
  monitor="accuracy"
  , patience=5
  , restore_best_weights=True
)

# search
num_epochs = 20
Io2_model.search(ds_train, epochs=num_epochs, callbacks=[es])


# ------------- get best/ fit ---------------
# best parameters
get_best_params(Io2_model, "drop_connect_rate", "learning_rate")

# best model summary 
Io2_best_model = get_best_model(Io2_model, ds_train)

# fit using the best model
Io2_best_model.fit(ds_train)


# ------------- results ---------------
# confusion_matrix, classification report
y_actual = []
for images, labels in ds_test:
    y_actual.append(np.argmax(labels.numpy(), axis=1))  # Extract labels from one-hot encoding
y_actual = np.concatenate(y_actual)

pred = Io2_best_model.predict(ds_test)
y_pred = np.argmax(pred, axis=1)

print("---- confusion matrix ----")
print(confusion_matrix(y_actual, y_pred))

print("---- classification report ----")
print(classification_report(y_actual, y_pred))



