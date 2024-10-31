import os
import glob
import time
import pickle
import pandas as pd
from picker import SeismicData
from obspy import Catalog, UTCDateTime, read_events
from obspy.core.event import Event as obspyEvent
from obspy.core.event import Origin, Pick, Magnitude, CreationInfo, QuantityError, ResourceIdentifier
from obspy.geodetics.base import gps2dist_azimuth

class GlobowCatalog(Catalog):
    def update_catalog(self, datalist: SeismicData, result_csv_filenames: list, version: str) -> None:
        count = 0; length = len(result_csv_filenames)
        for result_csv in map(pd.read_csv, result_csv_filenames):
            for index, row in result_csv.iterrows():
                event_id = ResourceIdentifier(id=f"evct-{row['file_name'].split('_')[-1]}")
                event = event_id.get_referred_object()
                if event is None:
                    table = datalist.get_event_by_centroid_time(UTCDateTime(row['file_name'].split('_')[-1]))
                    gcmt_id = ResourceIdentifier(id=f"gcmt{table.gcmtid}")
                    event = obspyEvent(resource_id=event_id,
                        origins=[Origin(
                            resource_id=gcmt_id,
                            time = table.srctime,
                            latitude = table.srcloc[0],
                            longitude = table.srcloc[1],
                            depth = table.srcloc[2],
                            )],
                        preferred_origin_id=f"gcmt{table.gcmtid}",
                        magnitudes=[Magnitude(mag=table.magnitude, magnitude_type='Mw', resource_id=ResourceIdentifier(id=f"gcmt{table.gcmtid}-Mw"))],
                        event_type="earthquake"
                        )
                    self.append(event)

                azimuths = gps2dist_azimuth(
                    lat1=row['station_lat'],
                    lon1=row['station_lon'],
                    lat2=event.preferred_origin().latitude,
                    lon2=event.preferred_origin().longitude
                    )
                
                for phase in ['P', 'S']:
                    pick_id = ResourceIdentifier(id=f"quakeml:jun.su/globowcat/{event_id}-{row['network']}_{row['station']}_LH-{phase}")
                    if pick_id.get_referred_object() is None:
                        if pd.isna(row[f'{phase.lower()}_arrival_time']): continue
                        event.picks.append(Pick(
                            resource_id=pick_id,
                            time=row[f'{phase.lower()}_arrival_time'],
                            time_errors=QuantityError(confidence_level=row[f'{phase.lower()}_probability']),
                            azimuth=azimuths[1],
                            backazimuth=azimuths[2],
                            method=ResourceIdentifier(id="globowcat-eqt-v5"),
                            phase_hint=phase,
                            creation_info = CreationInfo(
                                author="Jun Su", version=version, creation_time=UTCDateTime.now()
                                ),
                            ))
            count = count + 1
            print(f"Progress: [{count}/{length}] [{count/length*100:.1f}%] [{'='*int(count/length*20)}>{' '*(20-int(count/length*20))}]", end='\r')
        print(f"\nCatalog is updated for {length} result files.")
        return None

if __name__ == "__main__":
    old_version = "1.0-rc.1"
    new_version = "1.0-rc.1"
    old_filename_suffix = ""
    new_filename_subfix = ""
    datalist_dir = "/Users/junsu/Documents/data_gcmt.pkl"
    event_filter =  ""
    result_csv_filenames = glob.glob("./updeANMO_shift5_pred_catalog_*/*_outputs/X_prediction_results.csv")

    old_catalog_dir = f"./globowcat_{old_version}{old_filename_suffix}.xml"
    new_catalog_dir = f"./globowcat_{new_version}{new_filename_subfix}.xml"
    if not os.path.exists(old_catalog_dir):
        print("A new catalog will be created:", new_catalog_dir)
        globowcat = GlobowCatalog()
    else:
        start_time = time.time()
        print('Loading catalog:', old_catalog_dir, end="\r")
        globowcat = GlobowCatalog()
        globowcat.events = read_events(old_catalog_dir).events
        load_time = time.time() - start_time
        print(f"\nCatalog is loaded in {load_time:.1f} seconds.")

    globowcat.resource_id=str(f"quakeml:jun.su/globowcat_{new_version}")
    globowcat.creation_info=CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

    with open(datalist_dir, 'rb') as f: datalist = pickle.load(f)
    globowcat.update_catalog(datalist, result_csv_filenames, new_version)
    print('Saving file...', end="\r")
    globowcat.write(new_catalog_dir, format='QUAKEML')
    print("Catalog is saved:", new_catalog_dir)
