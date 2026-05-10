from eeg_engine import EEGEngine
from gradcam import GradCAM
import numpy as np
import tensorflow as tf

engine = EEGEngine()
engine.load_model()
model = engine.model
layer_name = 'conv1d_2'
conv_layer = model.get_layer(layer_name)

# Model 1
model_1 = tf.keras.models.Model(inputs=model.inputs, outputs=conv_layer.output)

# Model 2
# Start from the layer after conv_layer
layer_idx = model.layers.index(conv_layer)
x = tf.keras.Input(shape=conv_layer.output.shape[1:])
current = x
for layer in model.layers[layer_idx+1:]:
    current = layer(current)
model_2 = tf.keras.models.Model(inputs=x, outputs=current)

test_input = tf.convert_to_tensor(np.random.rand(1, 20, 1000).astype(np.float32))

with tf.GradientTape() as tape:
    conv_outputs = model_1(test_input)
    tape.watch(conv_outputs)
    predictions = model_2(conv_outputs)
    loss = predictions[:, 1]

grads = tape.gradient(loss, conv_outputs)
print("grads with TWO MODELS:", 'None' if grads is None else grads.shape)
