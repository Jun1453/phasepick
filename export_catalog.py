import os
import json
import glob
import time
import math
import pickle
import numpy as np
import pandas as pd
from cmcrameri import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm
from GeoPoint import GeoPoint, midpoint
from picker import SeismicData, flatten_list
from obspy import Catalog, UTCDateTime, read_events, read_inventory
from obspy.core.event import Event as obspyEvent
from obspy.core.event import Origin, Pick, Arrival, Magnitude, CreationInfo, QuantityError, ResourceIdentifier
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees

model = TauPyModel(model="prem")
prem_id = ResourceIdentifier(id=f"prem")
# ak135_id = ResourceIdentifier(id=f"ak135")

def get_station_dict(xml_path=None):
    if xml_path == None: xml_path = "rawdata_catalog_mass*_usta/*/*/stations/*.xml"
    inventory_dirs = glob.glob(xml_path)
    station_dict = {}
    for inventory_dir in inventory_dirs:
        key = f"{inventory_dir.split('/')[-1].split('.')[0]}.{inventory_dir.split('/')[-1].split('.')[1]}"
        if key in station_dict: continue
        station = read_inventory(inventory_dir)[0][0]
        station_dict[key] = {'latitude': station.latitude, 'longitude': station.longitude, 'elevation':station.elevation}
    return station_dict

def plot_depthslice(phase: str, value: str, gcarc_range: set, fidelity_func, value_constraint=lambda tb: pd.isnull(tb) | pd.notnull(tb), raw_filename = None, mark_size=lambda r: 50*r**2, colorscale=None, demean=False, summary_ray=False, plot_legend=False, plot_colorbar=True, plot=True):
    table = pd.concat([pd.read_pickle(filename) for filename in glob.glob(raw_filename)], ignore_index=True)
    picker_prob = table['probability']
    fidelity = fidelity_func(picker_prob)
    scatter_table = table[(table['phase'] == phase.upper()) &
                          (value_constraint(table[value])) &
                          (fidelity_func(picker_prob) > 0) &
                          (table['gcarc'] > gcarc_range[0]) &
                          (table['gcarc'] < gcarc_range[1])].copy()
    if summary_ray:
        count_org = len(scatter_table)
        std = np.std(scatter_table[value].values[:])
        half_length_norm = 2
        neighbor_table = table[(table['phase'] == phase.upper()) &
                               (value_constraint(table[value])) &
                               (fidelity_func(picker_prob) > 0) &
                               (table['gcarc'] > gcarc_range[0]-half_length_norm) &
                               (table['gcarc'] < gcarc_range[1]+half_length_norm)]
        for idx, rec in tqdm(scatter_table.iterrows(), total=len(scatter_table), desc="Processing rays"):
            half_length = 2
            while (True):
                neighbor_value = neighbor_table.loc[(abs(neighbor_table['turning_lon']-rec['turning_lon'])<half_length) &
                                                    (abs(neighbor_table['turning_lat']-rec['turning_lat'])<half_length) &
                                                    (abs(neighbor_table['gcarc']-rec['gcarc'])<half_length), value]
                if len(neighbor_value) > 200: half_length /= math.sqrt(2)
                else: break
            if len(neighbor_value) > 3:
                if (rec[value] > np.mean(neighbor_value) + std) or (rec[value] < np.mean(neighbor_value) - std):
                    table.loc[idx, picker_prob.name] = 0

        scatter_table = table[(table['phase'] == phase.upper()) &
                              (value_constraint(table[value])) &
                              (table['gcarc'] > gcarc_range[0]) &
                              (table['gcarc'] < gcarc_range[1])]
        scatter_table = scatter_table[fidelity_func(scatter_table[picker_prob.name]) > 0]
        print(f"{count_org-len(scatter_table)} outliers are removed")

    lons = scatter_table['turning_lon'].values[:]
    lats = scatter_table['turning_lat'].values[:]
    prob = scatter_table['probability'].values[:]
    anomaly = scatter_table[value].values[:]
    if demean: anomaly -= np.nanmean(anomaly)
    print(f"{sum(scatter_table[value].notnull())} points fit the conditions")

    if plot:
        map = Basemap(projection='moll',lon_0=-180,resolution='c')
        x, y = map(lons, lats)
        # anomaly -= np.nanmean(anomaly)

        map.drawmapboundary(fill_color='white')
        map.drawcoastlines()
        map.scatter(x, y, c=anomaly, s=mark_size(prob), marker='o', cmap=cm.vik)
        # map.scatter(x, y, c=anomaly, marker='o', cmap=cm.vik)
        if not colorscale:
            colorscale = 15 if phase.lower() == 's' else 8
        plt.clim(-colorscale, colorscale)
        # plt.title(f'{phase.upper()}-wave arrival time (sec) in {gcarc_range[0]}~{gcarc_range[1]} deg\n {value}, {sum(scatter_table[value].notnull())} pts')
        plt.title(f'{phase.upper()} travel time residuals (sec) in {gcarc_range[0]}~{gcarc_range[1]} deg\n{"demeaned" if demean else""} $\delta t{f"_{phase.upper()}" if value=="anomaly" else ""}$, {sum(scatter_table[value].notnull())} pts')

        if plot_colorbar:
            map.colorbar(location='bottom')

        if plot_legend:
            sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            markers = [plt.scatter([], [], s=mark_size(size), edgecolors='none', color='dimgray') for size in sizes]
            labels = [f'{size:.1f}' for size in sizes]
            plt.legend(markers, labels, title="probability", scatterpoints=1, frameon=False, loc='lower left', bbox_to_anchor=(0.97, 0.75), fontsize=12)

    return scatter_table

def plot_ratio(get_table1, get_table2, gcarc_range: set, xlim=[-20,20], ylim=[-20,20], demean=False, substract_coef=0, x_dep_y=False, slope_two_sections=False, x_label=None, y_label=None, title=None):
    table1 = get_table1(gcarc_range).rename(columns={'anomaly': 'anomaly1', 'probability': 'probability1'}) 
    table2 = get_table2(gcarc_range).rename(columns={'anomaly': 'anomaly2', 'probability': 'probability2'})
    table1['trace_id'] = [ "-".join(arrival_id.split("-")[:-2]) for arrival_id in table1['arrival_id'].values]
    table2['trace_id'] = [ "-".join(arrival_id.split("-")[:-2]) for arrival_id in table2['arrival_id'].values]
    columns_to_add = table2.columns.difference(table1.columns).append(pd.Index(['trace_id']))
    scatter_table = table1.merge(table2[columns_to_add], how='inner', on='trace_id')

    x = scatter_table[f'anomaly{1 if not x_dep_y else 2}'].values
    y = scatter_table[f'anomaly{2 if not x_dep_y else 1}'].values
    x_prob = scatter_table[f'probability{1 if not x_dep_y else 2}'].values
    y_prob = scatter_table[f'probability{2 if not x_dep_y else 1}'].values
    y -= substract_coef*x
    if demean:
        x -= np.mean(x)
        y -= np.mean(y)
        
    if not x_dep_y:
        plt.scatter(x, y, np.minimum(x_prob, y_prob))
    else:
        plt.scatter(y, x, np.minimum(x_prob, y_prob))

    # theilsen = TheilSenRegressor(random_state=42).fit(y.reshape(-1,1), x)
    # y_bound = np.array([min(y), max(y)])
    # # y_bound = np.array([-10,10]).reshape(-1,1)
    # b = round(float(np.diff(y_bound) / np.diff(theilsen.predict(y_bound.reshape(-1,1)))), 4)
    # plt.plot(theilsen.predict(y_bound.reshape(-1,1)), y_bound, color="k", lw=2)

    if slope_two_sections:
        slope = [None, None]
        x_bound = [x<=0, x>=0]
    else:
        slope = [None]
        x_bound = [x != np.nan]
    
    for i in range(len(slope)):
        fit_range = np.array([min(x[x_bound[i]]), max(x[x_bound[i]])])
        theilsen = TheilSenRegressor(random_state=42).fit(x[x_bound[i]].reshape(-1,1), y[x_bound[i]])
        slope[i] = round(float(np.diff(theilsen.predict(fit_range.reshape(-1,1))) / np.diff(fit_range)), 4)
        if not x_dep_y:
            if i == 0: plt.plot(np.array([min(x), max(x)]), theilsen.predict(np.array([min(x), max(x)]).reshape(-1,1)), color="darkgray", lw=2)
            plt.plot(fit_range, theilsen.predict(fit_range.reshape(-1,1)), color="k", lw=3)
            
        else:
            slope[i] = round(1/slope[i], 4)
            if i == 0: plt.plot(theilsen.predict(np.array([min(x), max(x)]).reshape(-1,1)), np.array([min(x), max(x)]), color="darkgray", lw=2)
            plt.plot(theilsen.predict(fit_range.reshape(-1,1)), fit_range, color="k", lw=3)
            


    # theilsen = TheilSenRegressor(random_state=42).fit(y[y<=0].reshape(-1,1), x[y<=0])
    # slope[0] = round(float(np.diff(y_fit_negative) / np.diff(theilsen.predict(y_fit_negative.reshape(-1,1)))), 4)
    # plt.plot(theilsen.predict(y_fit_negative.reshape(-1,1)), y_fit_negative, color="k", lw=3)
    # plt.plot(theilsen.predict(y_fit_positive.reshape(-1,1)), y_fit_positive, color="darkgray", lw=2)
    # theilsen = TheilSenRegressor(random_state=42).fit(y[y>=0].reshape(-1,1), x[y>=0])
    # slope[1] = round(float(np.diff(y_fit_positive) / np.diff(theilsen.predict(y_fit_positive.reshape(-1,1)))), 4)
    # plt.plot(theilsen.predict(y_fit_positive.reshape(-1,1)), y_fit_positive, color="k", lw=3)
    
    plt.xlim(xlim); plt.ylim(ylim)
    # print(plt.axes[0][0].get_ylim())
    if x_label is not False: plt.xlabel("$\delta t_P (sec)$" if x_label is None else x_label)
    if y_label is not False: plt.ylabel("$\delta t_S (sec)$" if y_label is None else y_label)
    plt.title((f'P and S travel time residuals in {gcarc_range[0]}~{gcarc_range[1]} deg\n' if title is None else title) + f'{len(x)} pts, ratio = {tuple(slope) if len(slope)>1 else slope[0]}' )
    
    return scatter_table


class GlobalCatalog(Catalog):
    def update_catalog(self, datalist: SeismicData, result_csv_filenames: list, version: str) -> None:
        count = 0; length = len(result_csv_filenames)
        for result_csv in map(pd.read_csv, result_csv_filenames):
            for index, row in result_csv.iterrows():
                event_id = ResourceIdentifier(id=f"quakeml:jun.su/globocat/evct-{row['file_name'].split('_')[-1]}")
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
                            creation_info = CreationInfo(author="Global CMT Project"),
                            )],
                        preferred_origin_id=f"gcmt{table.gcmtid}",
                        magnitudes=[Magnitude(mag=table.magnitude, magnitude_type='Mw', resource_id=ResourceIdentifier(id=f"gcmt{table.gcmtid}-Mw"))],
                        event_type="earthquake"
                        )
                    self.append(event)

                azimuths = gps2dist_azimuth(
                    lat1=event.preferred_origin().latitude,
                    lon1=event.preferred_origin().longitude,
                    lat2=row['station_lat'],
                    lon2=row['station_lon'],
                    )
                
                for phase in ['P', 'S']:
                    if (str(row['network']).lower() == 'nan') and (row['file_name'].split('_')[1] == 'NA'):
                        network = 'NA'
                    else: network = row['network']
                    pick_id = ResourceIdentifier(id=f"{event_id}-{str(network).strip()}_{str(row['station']).strip()}_LH-{phase}")
                    if pick_id.get_referred_object() is None:
                        if pd.isna(row[f'{phase.lower()}_arrival_time']): continue
                        event.picks.append(Pick(
                            resource_id=pick_id,
                            time=row[f'{phase.lower()}_arrival_time'],
                            time_errors=QuantityError(confidence_level=row[f'{phase.lower()}_probability']),
                            backazimuth=azimuths[2],
                            method=ResourceIdentifier(id="globocat-eqt-v5"),
                            phase_hint=phase,
                            creation_info = CreationInfo(
                                author="Jun Su", version=version, creation_time=UTCDateTime.now()
                                ),
                            ))
                        distance = kilometers2degrees(azimuths[0]/1000)
                        # print('distance:', distance)
                        prem_arrival_time = model.get_travel_times(event.preferred_origin().depth, distance, [phase])
                        if len(prem_arrival_time)>0:
                            event.preferred_origin().arrivals.append(Arrival(
                                resource_id=ResourceIdentifier(id=f"{pick_id.id}-PREM"),
                                earth_model_id=prem_id,
                                pick_id=pick_id,
                                phase=phase,
                                azimuth=azimuths[1],
                                distance=distance,
                                time_residual=(UTCDateTime(row[f'{phase.lower()}_arrival_time'])-event.preferred_origin().time)-prem_arrival_time[0].time,
                            ))
            count = count + 1
            print(f"Progress: [{count}/{length}] [{count/length*100:.1f}%] [{'='*int(count/length*20)}>{' '*(20-int(count/length*20))}]", end='\r')
        print(f"\nCatalog is updated for {length} result files.")
        return None
        
    def get_dataframe(self, include_id=False, xml_path=None, load_station_dict=None, reference_isc=False) -> pd.DataFrame:
        if not hasattr(self, 'station_dict'):
            if load_station_dict and os.path.exists(load_station_dict):
                with open(load_station_dict, 'r') as f:
                    self.station_dict = json.load(f)
            else:
                self.station_dict = get_station_dict(xml_path)
        df = pd.DataFrame([flatten_list([
                            arrival.resource_id.id.split('-')[-3].split('_')[:2],
                            arrival.phase,
                            round(arrival.time_residual, 2),
                            arrival.pick_id.get_referred_object().time_errors.confidence_level,
                            round(arrival.distance, 4),
                            round(arrival.azimuth, 2),
                            round(arrival.pick_id.get_referred_object().backazimuth, 2),
                            list(midpoint(GeoPoint(
                                    lat = ev.preferred_origin().latitude,
                                    lon = ev.preferred_origin().longitude
                                ), GeoPoint(
                                    lat = self.station_dict[f"{'.'.join(arrival.resource_id.id.split('-')[-3].split('_')[:2])}"]['latitude'],
                                    lon = self.station_dict[f"{'.'.join(arrival.resource_id.id.split('-')[-3].split('_')[:2])}"]['longitude']
                                )).get_latlon('deg', precision=3)
                            ),
                            arrival.resource_id.id
                        ]) for ev in self for arrival in ev.preferred_origin().arrivals ],
                    columns=['network', 'station', 'phase', 'anomaly', 'probability','gcarc', 'azimuth', 'backazimuth', 'turning_lat', 'turning_lon', 'arrival_id'],
                    )

        if reference_isc:
            def get_isc_event(isc_event_path):
                isc_events = read_events(isc_event_path)
                if len(isc_events) == 0: return None
                elif len(isc_events) > 1:raise Exception(f"More than one events in ISC catalog:", isc_event_path)
                else: isc_event = isc_events[0]
                if len(isc_event.origins) == 0: raise Exception(f"No origin in ISC catalog:", isc_event_path)
                elif len(isc_event.origins) > 1:raise Exception(f"More than one origins in ISC catalog:", isc_event_path)
                return isc_event
            
            for event in tqdm(globocat, desc="Processing events"):
                if UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6) > UTCDateTime('2022-10-31T23:59:59Z'): continue
                isc_p_event = get_isc_event(f"quakeml/P/{UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6)}.xml")
                isc_s_event = get_isc_event(f"quakeml/S/{UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6)}.xml")
                if isc_p_event: event.picks += isc_p_event.picks 
                if isc_s_event: event.picks += isc_s_event.picks 
                if isc_p_event:
                    if isc_s_event: isc_p_event.origins[0].arrivals += isc_s_event.origins[0].arrivals
                    event.origins.append(isc_p_event.origins[0])
                elif isc_s_event: event.origins.append(isc_s_event.origins[0])

            def find_related_isc_arrival(arrival_id):
                event_id, station_info, phase = arrival_id.rsplit('-', 3)[:-1]
                network_code, station_code = station_info.split('_')[:-1]
                event = ResourceIdentifier(id=event_id).get_referred_object()
                for origin in event.origins:
                    if origin.creation_info and origin.creation_info.author == 'ISC':
                        for isc_arrival in origin.arrivals:
                            if isc_arrival.phase == phase and isc_arrival.pick_id.get_referred_object().waveform_id.station_code == station_code:
                                return isc_arrival
                return None

            anomaly_isc = lambda arrival_id: find_related_isc_arrival(arrival_id).time_residual if find_related_isc_arrival(arrival_id) else np.NaN
            anomaly_rev = lambda arrival_id: ( anomaly_isc(arrival_id)
                                             + (ResourceIdentifier(arrival_id).get_referred_object().pick_id.get_referred_object().time
                                             - find_related_isc_arrival(arrival_id).pick_id.get_referred_object().time) ) if find_related_isc_arrival(arrival_id) else np.NaN

            df = pd.merge(df, pd.DataFrame([[
                        arrival_id,
                        round(anomaly_rev(arrival_id), 2),
                        anomaly_isc(arrival_id)
                    ] for arrival_id in tqdm(df['arrival_id'].values, desc="Processing arrivals")],
                columns=['arrival_id', 'anomaly_rev', 'anomaly_isc']
                ), on='arrival_id', how='inner')

        return df if include_id else df.drop('arrival_id', axis='columns')

if __name__ == "__main__":
    year = 2020
    old_version = "1.2.0"
    new_version = "1.2.0"
    old_filename_suffix = f"_{year}"
    new_filename_subfix = f"_{year}"
    datalist_dir = "/Users/junsu/Documents/data_gcmt.pkl"
    event_filter = ""
    result_csv_filenames = glob.glob(f"./updeANMO_shift5_pred_catalog_*{year}/*_outputs/X_prediction_results.csv")

    old_catalog_dir = f"./globocat_{old_version}{old_filename_suffix}.xml"
    new_catalog_dir = f"./globocat_{new_version}{new_filename_subfix}.xml"
    if not os.path.exists(old_catalog_dir):
        print("A new catalog will be created:", new_catalog_dir)
        globocat = GlobalCatalog()
    else:
        start_time = time.time()
        print('Loading catalog:', old_catalog_dir, end="\r")
        globocat = GlobalCatalog()
        globocat.events = read_events(old_catalog_dir).events
        load_time = time.time() - start_time
        print(f"\nCatalog is loaded in {load_time:.1f} seconds.")

    globocat.resource_id=str(f"quakeml:jun.su/globocat_{new_version}")
    globocat.creation_info=CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

    with open(datalist_dir, 'rb') as f: datalist = pickle.load(f)
    globocat.update_catalog(datalist, result_csv_filenames, new_version)
    print('Saving file...', end="\r")
    globocat.write(new_catalog_dir, format='QUAKEML')
    print("Catalog is saved:", new_catalog_dir)

    df = globocat.get_dataframe(load_station_dict="station_dict.json", reference_isc=True, include_id=True)
    df.to_pickle(f'updeANMO_shift5_catalog{new_filename_subfix}_plot.pkl')
