# use virtual environment venv-3.7
import numpy as np
import EQTransformer as eqt
from EQTransformer.core.predictor import predictor
# from EQTransformer.core.predictor_sfocus import predictor_sfocus
predictor(input_dir= '/Volumes/seismic/catalog_hdfs', input_model='test_trainer_updeANMO_shift5_outputs/final_model.h5',
          output_dir='/Volumes/seismic/catalog_hdfs/updeANMO_shift5_pred_catalog', detection_threshold=0.3, P_threshold=0.1,
          S_threshold=0.1, number_of_plots=10, plot_mode='time')