from eeg_engine import EEGEngine
from gradcam import GradCAM

engine = EEGEngine()
engine.load_model()
try:
    print("try model.outputs:", getattr(engine.model, 'outputs', 'NONE'))
except Exception as e:
    print("outputs err:", e)

try:
    print("try model.output:", engine.model.output)
except Exception as e:
    print("output err:", e)
