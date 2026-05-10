import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessor import EEGPreprocessor

pre = EEGPreprocessor(500)
rel_files = ['subset_train/Subject00_1.edf', 'subset_train/Subject01_1.edf']
st_files = ['subset_train/Subject00_2.edf', 'subset_train/Subject01_2.edf']

print('Running preprocessing smoke test on subset_train/...')

# process relaxed files
rel_segs = []
for f in rel_files:
    segs, ch = pre.preprocess_pipeline(f, apply_ica=False, segment_length=2.0, overlap=1.0)
    segs = pre.extract_first_20_channels(segs)
    rel_segs.extend(segs)

# process stressed files
st_segs = []
for f in st_files:
    segs, ch = pre.preprocess_pipeline(f, apply_ica=False, segment_length=2.0, overlap=1.0)
    segs = pre.extract_first_20_channels(segs)
    st_segs.extend(segs)

print(f"Relaxed segments: {len(rel_segs)}")
print(f"Stressed segments: {len(st_segs)}")
print(f"Total segments: {len(rel_segs) + len(st_segs)}")
print('Example segment shape:', rel_segs[0].shape)
