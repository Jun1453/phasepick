# use virtual environment venv-3.7
import sys
import numpy as np
import EQTransformer as eqt
from EQTransformer.core.predictor import predictor
# from EQTransformer.core.predictor_sfocus import predictor_sfocus
if not (int(sys.argv[1]) > 1970 and int(sys.argv[1]) < 2100): raise ValueError("invalid args")
predictor(input_dir= f"/Volumes/seismic/catalog_{sys.argv[1]}_stnflt_hdfs", input_model='test_trainer_updeANMO_shift5_outputs/final_model.h5',
          output_dir=f'/Volumes/seismic/catalog_{sys.argv[1]}_stnflt_hdfs/updeANMO_shift5_pred_catalog', detection_threshold=0.3, P_threshold=0.1,
          S_threshold=0.1, number_of_plots=10, plot_mode='time')