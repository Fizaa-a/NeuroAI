from eeg_engine import EEGEngine
from gradcam import GradCAM

engine = EEGEngine()
engine.load_model()
try:
    gc = GradCAM(engine.model)
    print("Success building GradCAM")
except Exception as e:
    print("Error building GradCAM:", e)
