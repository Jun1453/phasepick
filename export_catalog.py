import os
import glob
import pickle
import pandas as pd
from picker import SeismicData
from obspy import Catalog, UTCDateTime, read_events
from obspy.core.event import Event as obspyEvent
from obspy.core.event import Origin, Pick, Magnitude, CreationInfo, QuantityError, ResourceIdentifier
from obspy.geodetics.base import gps2dist_azimuth

def update_catalog(catalog: Catalog, datalist: SeismicData, result_csv_filenames: list, version: str) -> Catalog:
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
                    magnitudes=[Magnitude(mag=table.magnitude, magnitude_type='Mw', resource_id=gcmt_id)],
                    event_type="earthquake"
                    )
                catalog.append(event)

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
        print(f"progress: {count}/{length}")

if __name__ == "__main__":
    old_version = "1.0-b.2"
    new_version = "1.0-rc.1"
    datalist_dir = "/Users/junsu/Documents/data_gcmt.pkl"
    result_csv_filenames = glob.glob("./updeANMO_shift5_pred_catalog_*/*_outputs/X_prediction_results.csv")

    if not os.path.exists(f"./globowcat_{old_version}.xml"):
        globowcat = Catalog()
    else:
        globowcat = read_events(f"./globowcat_{old_version}.xml")

    globowcat.resource_id=str(f"quakeml:jun.su/globowcat_{new_version}")
    globowcat.creation_info=CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

    with open(datalist_dir, 'rb') as f: datalist = pickle.load(f)
    update_catalog(globowcat, datalist, result_csv_filenames, new_version)
    globowcat.write(f"./globowcat_{new_version}.xml", format='QUAKEML')
