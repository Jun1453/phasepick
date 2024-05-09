YEAR=2013
source /Users/jun/.local/share/virtualenvs/phasepick-mEodVLkE/bin/activate
/usr/local/Cellar/rsync/3.3.0/bin/rsync -avuh "basanos:/volume1/seismic/rawdata_catalog_mass/${YEAR}*" ./rawdata_catalog_mass
python picker.py ${YEAR} 2>&1 | tee prepare_e2e_${YEAR}.log
/usr/local/Cellar/rsync/3.3.0/bin/rsync -avuh ./training_catalog_mass/* /Volumes/seismic/training_catalog_mass
/bin/rm -rf ./rawdata_catalog_mass/* ./training_catalog_mass/*
source venv-3.7/bin/activate
python predict.py ${YEAR} 2>&1 | tee predict_e2e_${YEAR}.log
