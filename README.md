# Machine Learning Phase Picker for Long-period Teleseismic Wave Arrivals

This repository contains the machine learning phase picker code used to generate GLoBoCat (Global Long-period Body Wave Catalog). The published picker (Su+2025) is trained on onset arrivals picked by experts (Houser+2008) and enables automated detection of P and S wave arrivals in long-period teleseismic data. Here we provide the original scripts to train and apply the picker.

The generated GLoBoCat dataset comprises ~2 million long period P and S arrivals for earthquakes M>5.5 based on Global CMT and stations available on IRIS Data Management Center between 2010 and 2023. This work is submitted to Geophysical Research Letters for peer-review.

## Generated Dataset Components

The picker generates GLoBoCat in multiple formats:
- **ObsPy format** (`obspy_catalog.tar.gz`): Contains hierarchical objects Event, Origin, Pick, and Arrival for cross-reference
- **Pandas format** (`pandas_table.tar.gz`): Simplified information focused on arrivals for rapid analysis
- **Text format** (`text_table.tar.gz`): Plain text format for easy parsing
- **Trained model** (`final_model.h5`): The picking model that can be loaded with EQTransformer (Mousavi+2020)

## File Structure

### Core Python Modules

- **`picker.py`** - Main seismic data processing and picking module
  - `SeismicData` class: Handles seismic data management, fetching, and preprocessing
  - `Picker` class: Main interface for phase picking operations
  - `Event`, `Station`, `Record` classes: Data structures for seismic events
  - `Instrument` class: Handles instrument response management

- **`predict.py`** - EQTransformer prediction script for automated phase picking
  - Uses trained model to predict P and S wave arrivals
  - Processes seismic waveforms and outputs arrival time predictions

- **`GeoPoint.py`** - Geographic coordinate handling utilities
  - `GeoPoint` class: Manages latitude/longitude coordinates
  - `midpoint` function: Calculates geographic midpoints

### Data Export and Analysis

- **`export_catalog.py`** - Catalog export and visualization tools
  - `GlobalCatalog` class: Manages global seismic catalogs
  - Functions for plotting depth slices and travel time residuals
  - Integration with ISC (International Seismological Centre) data

- **`export_isc_mp.py`** - Multi-processing version of catalog export
  - Parallel processing for large-scale catalog operations

- **`stalist.py`** - Station list generation utility
  - Creates station inventory from seismic data files

### Analysis and Visualization

- **`fig4-prep_table.py`** - Figure preparation utilities
- **`plot_vpvs.py`** - Vp/Vs ratio plotting tools
- **`mc_uncert.py`** - Magnitude and uncertainty analysis
- **`ttresmap-interactive.py`** - Interactive travel time residual mapping

### Data Files

- **`gcmt_mw.npy`** - Global CMT catalog with moment magnitudes
- **`requirement.txt`** - Python package dependencies
- **`Pipfile`** and **`Pipfile.lock`** - Pipenv dependency management

### Shell Scripts

- **`preproc_at_ssd.sh`** - Preprocessing script for SSD storage

## Usage

### Prerequisites

Install required dependencies:
```bash
pip install -r requirement.txt
```

Or using pipenv:
```bash
pipenv install
```

### Basic Workflow

1. **Data Preparation**
   ```python
   from picker import Picker
   
   # Create picker instance
   picker = Picker(dataset_paths, dataset_as_folder=False)
   
   # Load existing dataset
   picker.load_dataset('data_fetched.pkl', verbose=True)
   ```

2. **Phase Picking**
   ```python
   # Prepare catalog for prediction
   picker.prepare_catalog(
       waveform_dir='./training_data',
       preproc_dir='./preproc',
       save_dir='./hdfs_output',
       n_processor=10
   )
   
   # Run prediction using EQTransformer
   python predict.py [year]
   ```

3. **Generate GLoBoCat**
   ```python
   from export_catalog import GlobalCatalog
   
   # Create and update catalog
   catalog = GlobalCatalog()
   catalog.update_catalog(datalist, result_csv_filenames, version)
   
   # Export to various formats
   df = catalog.get_dataframe(include_id=True)
   ```

### Data Processing Pipeline

1. **Fetch seismic data** from IRIS DMC
2. **Preprocess waveforms** (instrument response removal, filtering)
3. **Train/apply ML model** for phase picking
4. **Generate GLoBoCat** in multiple formats (ObsPy, Pandas, text)
5. **Export results** for analysis and visualization

### Key Features

- **Multi-format support**: ObsPy, Pandas, and plain text outputs
- **Parallel processing**: Efficient handling of large datasets
- **Quality control**: Built-in filtering and validation
- **Visualization tools**: Maps, plots, and interactive displays
- **ISC integration**: Comparison with International Seismological Centre data

## Generated Data Formats

The phase picker generates GLoBoCat in multiple formats:

### ObsPy Format
- Hierarchical Event/Origin/Pick/Arrival structure
- Full metadata preservation
- Compatible with ObsPy ecosystem

### Pandas Format
- Fast loading and analysis
- Tabular data structure
- Easy integration with data science workflows

### Text Format
- Human-readable
- Simple parsing
- Lightweight storage

## Citation

If you use this dataset, please cite:
```
J. Su, C. Houser, & J. W. Hernlund (2025). Global Long-wavelength Body-wave Catalog (GLoBoCat) for Lower Mantle P and S Wave Travel Times. ESS Open Archive. May 23, 2025.
```

## License

This repository is shared under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE.md).

## Contact

For questions about this dataset, please contact the corresponding author of Su+2025.
