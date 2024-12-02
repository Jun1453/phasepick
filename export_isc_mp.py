from export_catalog import *

from multiprocessing import Pool
import os
import glob
import time
import pickle
from obspy import UTCDateTime
from obspy.core.event import CreationInfo

# Main execution
if __name__ == '__main__':
    old_version = "1.2.0"
    new_version = "1.2.0"
    datalist_dir = "/Users/junsu/Documents/data_gcmt.pkl"
    event_filter = ""

    def process_year(year):
        old_filename_suffix = f"_{year}"
        new_filename_subfix = old_filename_suffix
        result_csv_filenames = glob.glob(f"./updeANMO_shift5_pred_catalog_*{year}/*_outputs/X_prediction_results.csv")

        old_catalog_dir = f"./globocat_{old_version}{old_filename_suffix}.xml"
        new_catalog_dir = f"./globocat_{new_version}{new_filename_subfix}.xml"
        
        if not os.path.exists(old_catalog_dir):
            print(f"A new catalog will be created for {year}: {new_catalog_dir}")
            globocat = GlobalCatalog()
        else:
            start_time = time.time()
            print(f'Loading catalog for {year}: {old_catalog_dir}')
            globocat = GlobalCatalog()
            globocat.events = read_events(old_catalog_dir).events
            load_time = time.time() - start_time
            print(f"Catalog for {year} is loaded in {load_time:.1f} seconds.")

        globocat.resource_id = str(f"quakeml:jun.su/globocat_{new_version}")
        globocat.creation_info = CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

        with open(datalist_dir, 'rb') as f:
            datalist = pickle.load(f)
        
        globocat.update_catalog(datalist, result_csv_filenames, new_version)
        print(f'Saving file for {year}...')
        globocat.write(new_catalog_dir, format='QUAKEML')
        print(f"Catalog is saved for {year}: {new_catalog_dir}")

        df = globocat.get_dataframe(load_station_dict="station_dict.json", reference_isc=True, include_id=True)
        df.to_pickle(f'updeANMO_shift5_catalog{new_filename_subfix}_plot.pkl')
        
        return year

    years = range(2010, 2023)
    
    # # Create a pool of workers
    # with Pool(processes=7) as pool:
    #     # Map the process_year function to all years
    #     results = pool.map(process_year, years)
    for year in years:
        process_year(year)
    
    print("All years processed successfully:", results)


# if __name__ == "__main__":

#     for year in range(2010, 2023):
#         old_filename_suffix = f"_{year}"
#         new_filename_subfix = old_filename_suffix
#         result_csv_filenames = glob.glob(f"./updeANMO_shift5_pred_catalog_*{year}/*_outputs/X_prediction_results.csv")

#         old_catalog_dir = f"./globocat_{old_version}{old_filename_suffix}.xml"
#         new_catalog_dir = f"./globocat_{new_version}{new_filename_subfix}.xml"
#         if not os.path.exists(old_catalog_dir):
#             print("A new catalog will be created:", new_catalog_dir)
#             globocat = GlobalCatalog()
#         else:
#             start_time = time.time()
#             print('Loading catalog:', old_catalog_dir, end="\r")
#             globocat = GlobalCatalog()
#             globocat.events = read_events(old_catalog_dir).events
#             load_time = time.time() - start_time
#             print(f"\nCatalog is loaded in {load_time:.1f} seconds.")

#         globocat.resource_id=str(f"quakeml:jun.su/globocat_{new_version}")
#         globocat.creation_info=CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

#         with open(datalist_dir, 'rb') as f: datalist = pickle.load(f)
#         globocat.update_catalog(datalist, result_csv_filenames, new_version)
#         print('Saving file...', end="\r")
#         globocat.write(new_catalog_dir, format='QUAKEML')
#         print("Catalog is saved:", new_catalog_dir)

#         df = globocat.get_dataframe(load_station_dict="station_dict.json", reference_isc=True, include_id=True)
#         df.to_pickle(f'updeANMO_shift5_catalog{new_filename_subfix}_plot.pkl')
