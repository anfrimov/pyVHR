import pyVHR as vhr
import numpy as np
from pyVHR.analysis.pipeline import Pipeline, DeepPipeline
from pathlib import Path
from pyVHR.utils.errors import getErrors, printErrors, displayErrors
import pickle

# params
wsize = 0
roi_approach = 'holistic'   # 'holistic' or 'patches'
bpm_est = 'median'         # BPM final estimate, if patches choose 'median' or 'clustering'
method = 'cpu_SSR'      # one of the methods implemented in pyVHR
# methods = ['cupy_CHROM', 'cpu_LGI', 'cupy_POS', 'cpu_PBV', 'cpu_PCA', 'cpu_GREEN', 'cpu_OMIT', 'cpu_ICA']
methods = ['cupy_CHROM', 'cupy_POS', 'cpu_LGI']
pre_filt = True
pipe = Pipeline()          # object to execute the pipeline

videoFileName = str(Path(r"C:\Users\MBEGroup\Documents\mbegroup\TeMoRett\data\rppg-eval\sourcedata\sub-666\Camera0_1.avi"))
#fps = vhr.extraction.get_fps(videoFileName)

cfg = { 
    'videoFileName': videoFileName,
    'winsize': wsize, 
    'roi_method': 'convexhull',
    'roi_approach': roi_approach,
    'methods': methods,
    'estimate': bpm_est,
    'patch_size': 40, 
    'RGB_LOW_HIGH_TH': (5,230),
    'Skin_LOW_HIGH_TH': (5,230),
    'pre_filt': pre_filt,
    'post_filt': True,
    'cuda': True, 
    'verb': True
}

# run
bvps, timesES, bpmES = pipe.run_on_video_multimethods(**cfg)

output = {'config': cfg, 'bvps': bvps, 'timesES': timesES, 'bpmES': bpmES}

with open(f"output_elias_{roi_approach}_win-{wsize}_pre-{pre_filt}.pkl", "wb") as f:
    pickle.dump(output, f)

print('done')
import winsound
winsound.Beep(440, 500)

# %% Deep Learning Pipeline

# dpipe = DeepPipeline()
# methods = ['MTTS_CAN', 'HR_CNN', 'TRANSFORMER']

# for m in methods:
#     print(f'Starting {m}...')
#     bvps, timesES, bpmES, median_bpmES = dpipe.run_on_video(videoFileName, wsize = 8, cuda=True, method=m, bpm_type='welch', post_filt=False, verb=True)

#     output = {'bvps': bvps, 'timesES': timesES, 'bpmES': bpmES, 'median_bpmES': median_bpmES}

#     with open(f"output_0_{m}.pkl", "wb") as f:
#         pickle.dump(output, f)

#     print(f'Done with {m}.')
