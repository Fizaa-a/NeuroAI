from eeg_engine import EEGEngine
import numpy as np

engine = EEGEngine()
engine.load_model()
model = engine.model

print("Trying model.inputs:", getattr(model, 'inputs', 'NONE'))
print("Trying model.input directly:", end=" ")
try:
    print(model.input)
except Exception as e:
    print("ERROR:", e)

print("Trying model.layers[0].input:", end=" ")
try:
    print(model.layers[0].input)
except Exception as e:
    print("ERROR:", e)

print("Trying model.layers[0].input_shape:", end=" ")
try:
    print(model.layers[0].input_shape)
except Exception as e:
    print("ERROR:", e)
