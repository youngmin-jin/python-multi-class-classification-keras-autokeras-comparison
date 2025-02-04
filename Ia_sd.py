import numpy as np
import tensorflow_datasets as tfds
import autokeras as ak
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ------------ autokeras ------------
# Load the Stanford Dogs dataset
builder = tfds.builder("stanford_dogs")
builder.download_and_prepare()

ds_train = builder.as_dataset(split='train')
ds_test = builder.as_dataset(split='test')

# unpacked and resize
def unpack_and_resize(example):
  image = example['image']
  label = example['label']
  
  image = tf.image.resize(image, [224,224])
  return image, label

# apply the unpacked func
ds_train = ds_train.map(unpack_and_resize)
ds_test = ds_test.map(unpack_and_resize)


# Initialize an AutoKeras ImageClassifier
Ia_model = ak.ImageClassifier(metrics=['accuracy'], max_trials=20, overwrite=True)

# Train the model
num_epochs = 30
Ia_model.fit(ds_train, epochs=num_epochs)

# summary 
Ia_model_result = Ia_model.export_model()
print(Ia_model_result.summary())

# confusion_matrix, classification report
y_actual = []
for image, label in ds_test:
  y_actual.append(label.numpy())

pred = Ia_model.predict(ds_test)
y_pred = pred.flatten().astype('int')

print("---- confusion matrix ----")
print(confusion_matrix(y_actual, y_pred))

print("---- classification report ----")
print(classification_report(y_actual, y_pred))

print("-------- specific parameters ----------")
for i, layer in enumerate(Ia_model_result.layers):
  print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

  # Get all layer attributes and print them
  config = layer.get_config()
  print("Layer Configuration:")
  for key, value in config.items():
    print(f"  {key}: {value}")

  # If the layer has weights, print their shapes
  if layer.weights:
    print("Layer Weights:")
    for weight in layer.weights:
      print(f"  {weight.name}: shape={weight.shape}")

  print("-" * 50)


