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

def plot_depthslice(phase: str, value: str, gcarc_range: set, fidelity_func, value_constraint=lambda tb: pd.isnull(tb) | pd.notnull(tb), raw_filename = None, mark_size=lambda r: 50*r**2, colorscale=None, demean=False, summary_ray=False, plot_legend=False, plot_colorbar=True, plot=True, include_id=False):
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

    return scatter_table if include_id else scatter_table.drop(['arrival_id', 'event_id', 'trace_id'], axis='columns')

def plot_ratio(table1, table2, gcarc_range: set, xlim=[-20,20], ylim=[-20,20], demean=False, substract_coef=0, x_dep_y=False, slope_two_sections=False, x_label=None, y_label=None, title=None, suffix1='1', suffix2='2', include_id=False):
    table1 = table1.rename(columns={'anomaly': f'anomaly{suffix1}', 'probability': f'probability{suffix1}'}) 
    table2 = table2.rename(columns={'anomaly': f'anomaly{suffix2}', 'probability': f'probability{suffix2}'})
    table1['trace_id'] = [ arrival_id.rsplit("/",2)[0] for arrival_id in table1['arrival_id'].values]
    table2['trace_id'] = [ arrival_id.rsplit("/",2)[0] for arrival_id in table2['arrival_id'].values]
    columns_to_add = table2.columns.difference(table1.columns).append(pd.Index(['trace_id']))
    scatter_table = table1.merge(table2[columns_to_add], how='inner', on='trace_id')

    x = scatter_table[f'anomaly{suffix1 if not x_dep_y else suffix2}'].values
    y = scatter_table[f'anomaly{suffix2 if not x_dep_y else suffix1}'].values
    x_prob = scatter_table[f'probability{suffix1 if not x_dep_y else suffix2}'].values
    y_prob = scatter_table[f'probability{suffix2 if not x_dep_y else suffix1}'].values
    y -= substract_coef*x
    if demean:
        x -= np.mean(x)
        y -= np.mean(y)
        
    if not x_dep_y:
        plt.scatter(x, y, np.minimum(x_prob, y_prob))
    else:
        plt.scatter(y, x, np.minimum(x_prob, y_prob))

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
            plt.plot(fit_range, theilsen.predict(fit_range.reshape(-1,1)), color="w" if plt.rcParams["figure.facecolor"] == 'black' else "k", lw=3)
            
        else:
            slope[i] = round(1/slope[i], 4)
            if i == 0: plt.plot(theilsen.predict(np.array([min(x), max(x)]).reshape(-1,1)), np.array([min(x), max(x)]), color="darkgray", lw=2)
            plt.plot(theilsen.predict(fit_range.reshape(-1,1)), fit_range, color="w" if plt.rcParams["figure.facecolor"] == 'black' else "k", lw=3)
            
    plt.xlim(xlim); plt.ylim(ylim)
    # print(plt.axes[0][0].get_ylim())
    if x_label is not False: plt.xlabel("$\delta t_P (sec)$" if x_label is None else x_label)
    if y_label is not False: plt.ylabel("$\delta t_S (sec)$" if y_label is None else y_label)
    plt.title((f'P and S travel time residuals in {gcarc_range[0]}~{gcarc_range[1]} deg\n' if title is None else title) + f'{len(x)} pts, ratio = {tuple(slope) if len(slope)>1 else slope[0]}' )
    
    return scatter_table if include_id else scatter_table.drop(['arrival_id', 'event_id', 'trace_id'], axis='columns')


class GlobalCatalog(Catalog):
    def update_catalog(self, datalist: SeismicData, result_csv_filenames: list, version: str, month=None) -> None:
        # get the whole table
        source_receiver = pd.concat([pd.read_csv(filename, keep_default_na=False) for filename in result_csv_filenames], ignore_index=True)
        source_receiver['network'] = source_receiver['network'].str.strip()
        source_receiver['station'] = source_receiver['station'].str.strip()
        srctimes = source_receiver['file_name'].str.split('_',n=-1).str[-1]
        if month:
            source_receiver = source_receiver[pd.to_datetime(srctimes).dt.month == month]
            srctimes = source_receiver['file_name'].str.split('_',n=-1).str[-1]
        gcmt_cats = [read_events(f'gcmt_data/{yr}/{UTCDateTime(yr, mon+1, 1).strftime("%b%y").lower()}.ndk') for yr in pd.to_datetime(srctimes).dt.year.unique() for mon in (range(12) if not month else [month])]
        
        # get unique events
        def find_event(gcmtid):
            event = None
            id_candidates = [f"smi:local/ndk/{letter}{gcmtid[1]}/event" for letter in ["C", "S", "M", "B"]]
            while (event is None):
                gcmtev_id = ResourceIdentifier(id=id_candidates.pop(0))
                event = gcmtev_id.get_referred_object()
            event.resource_id = ResourceIdentifier(id=f"quakeml:jun.su/globocat/evct-{gcmtid[0]}")
            return event
        
        get_gcmtid = lambda t: (t, datalist.get_event_by_centroid_time(UTCDateTime(t)).gcmtid)
        events = [find_event(get_gcmtid(srctime)) for srctime in srctimes.unique()]

        # get azimuth and distance using preferred origin
        srctime_to_refrrred_object = lambda t: ResourceIdentifier(id=f"quakeml:jun.su/globocat/evct-{t}").get_referred_object()
        source_receiver['event_ref'] = srctimes.apply(srctime_to_refrrred_object)
        source_receiver['azimuths'] = source_receiver.apply(lambda row: gps2dist_azimuth(
                                        lat1=row['event_ref'].preferred_origin().latitude,
                                        lon1=row['event_ref'].preferred_origin().longitude,
                                        lat2=row['station_lat'],
                                        lon2=row['station_lon'],
                                        ), axis=1)
        source_receiver['distance'] = source_receiver['azimuths'].apply(lambda x: kilometers2degrees(x[0]/1000))

        # get picks and arrivals
        def get_picks(phase, row):
            # print(row.name, row[f'file_name'])
            row['event_ref'].picks.append(Pick(
                resource_id=row['pickid'],
                time=row[f'{phase.lower()}_arrival_time'],
                time_errors=QuantityError(confidence_level=row[f'{phase.lower()}_probability']),
                backazimuth=row['azimuths'][2],
                method=ResourceIdentifier(id="globocat-eqt-v5"),
                phase_hint=phase,
                creation_info = CreationInfo(
                    author="Jun Su", version=version, creation_time=UTCDateTime.now()
                    ),
                ))

            for origin in row['event_ref'].origins:
                if origin == row['event_ref'].preferred_origin():
                    azimuths = row['azimuths']
                    distance = row['distance']
                else:
                    azimuths = gps2dist_azimuth(
                                lat1=origin.latitude,
                                lon1=origin.longitude,
                                lat2=row['station_lat'],
                                lon2=row['station_lon'],
                                )
                    distance = kilometers2degrees(azimuths[0]/1000)
                prem_arrival_time = model.get_travel_times(origin.depth/1000, distance, [phase])
                
                if len(prem_arrival_time)>0:
                    origin.arrivals.append(Arrival(
                        resource_id=ResourceIdentifier(id=f"{origin.resource_id.id}/{row['network']}_{row['station']}_LH/{phase}/PREM"),
                        earth_model_id=prem_id,
                        pick_id=row['pickid'],
                        phase=phase,
                        azimuth=azimuths[1],
                        distance=distance,
                        time_residual=(UTCDateTime(row[f'{phase.lower()}_arrival_time'])-origin.time)-prem_arrival_time[0].time,
                    ))
                    
        for phase in ['P', 'S']:
            arrival = source_receiver[source_receiver[f'{phase.lower()}_arrival_time']!='']
            #{row['event_id']}/{str().strip()}_{str(row['station']).strip()}_LH/{phase}"
            arrival['pickid'] = (arrival['event_ref'].apply(lambda x: x.resource_id.id)\
                                +'/'+arrival['network']+'_'+arrival['station']\
                                +f'_LH/{phase}').apply(ResourceIdentifier)
            # .apply(ResourceIdentifier(id=f"))                    
            
            tqdm.pandas(desc=phase)
            arrival.progress_apply(lambda row: get_picks(phase, row), axis=1)


        # update catalog
        for event in events:
            self.append(event)

        del events, gcmt_cats, source_receiver, arrival
        return None
        
    def get_dataframe(self, include_id=False, xml_path=None, load_station_dict=None, reference_isc=False, alt_origin=None) -> pd.DataFrame:
        origin = lambda ev: ResourceIdentifier(id=f"{ev.preferred_origin().resource_id.id.rsplit('/',1)[0]}/{alt_origin}").get_referred_object() if alt_origin else ev.preferred_origin()
        if not hasattr(self, 'station_dict'):
            if load_station_dict and os.path.exists(load_station_dict):
                with open(load_station_dict, 'r') as f:
                    self.station_dict = json.load(f)
            else:
                self.station_dict = get_station_dict(xml_path)
        df = pd.DataFrame([flatten_list([
                            arrival.resource_id.id.split('/')[-3].split('_')[:2],
                            arrival.phase,
                            arrival.pick_id.get_referred_object().time,
                            ev.resource_id.id.rsplit('/',1)[1][5:],
                            round(UTCDateTime(arrival.pick_id.get_referred_object().time)-UTCDateTime(ev.resource_id.id.rsplit('/',1)[1][5:]),2),
                            round(arrival.time_residual, 2),
                            arrival.pick_id.get_referred_object().time_errors.confidence_level,
                            round(arrival.distance, 4),
                            round(arrival.azimuth, 2),
                            round(arrival.pick_id.get_referred_object().backazimuth, 2) if arrival.pick_id.get_referred_object().backazimuth else None,
                            round(origin(ev).latitude, 3),
                            round(origin(ev).longitude, 3),
                            round(origin(ev).depth/1000, 3),
                            list(midpoint(GeoPoint(
                                    lat = origin(ev).latitude,
                                    lon = origin(ev).longitude
                                ), GeoPoint(
                                    lat = self.station_dict[f"{'.'.join(arrival.resource_id.id.split('/')[-3].split('_')[:2])}"]['latitude'],
                                    lon = self.station_dict[f"{'.'.join(arrival.resource_id.id.split('/')[-3].split('_')[:2])}"]['longitude']
                                )).get_latlon('deg', precision=3)
                            ),
                            round(self.station_dict[f"{'.'.join(arrival.resource_id.id.split('/')[-3].split('_')[:2])}"]['latitude'], 3),
                            round(self.station_dict[f"{'.'.join(arrival.resource_id.id.split('/')[-3].split('_')[:2])}"]['longitude'], 3),
                            arrival.resource_id.id,
                            ev.resource_id.id
                        ]) for ev in self for arrival in origin(ev).arrivals ],
                    columns=['network', 'station', 'phase', 'arrival_time', 'origin_time', 'travel_time', 'anomaly', 'probability','gcarc', 'azimuth', 'backazimuth', 'origin_lat', 'origin_lon', 'origin_dep', 'turning_lat', 'turning_lon', 'station_lat', 'station_lon', 'arrival_id', 'event_id'],
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
            
            for event in tqdm(self, desc="Processing events"):
                if UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6) > UTCDateTime('2022-10-31T23:59:59Z'): continue
                isc_p_event = get_isc_event(f"quakeml/P/{UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6)}.xml")
                isc_s_event = get_isc_event(f"quakeml/S/{UTCDateTime('-'.join(event.resource_id.id.split('-')[1:]), precision=6)}.xml")
                if isc_p_event: event.picks += isc_p_event.picks 
                if isc_s_event: event.picks += isc_s_event.picks 
                if isc_p_event:
                    if isc_s_event: isc_p_event.origins[0].arrivals += isc_s_event.origins[0].arrivals
                    event.origins.append(isc_p_event.origins[0])
                elif isc_s_event: event.origins.append(isc_s_event.origins[0])

            def find_related_isc_arrival(ids):
                arrival_id, event_id = ids
                origin_id, station_info, phase = arrival_id.rsplit('/', 3)[:-1]
                #event_id = f"quakeml:jun.su/globocat/evct-{ResourceIdentifier(origin_id).get_referred_object().time}"
                network_code, station_code = station_info.split('_')[:-1]
                # print(event_id)
                event = ResourceIdentifier(id=event_id).get_referred_object()
                for origin in event.origins:
                    if origin.creation_info and origin.creation_info.author == 'ISC':
                        for isc_arrival in origin.arrivals:
                            if isc_arrival.phase == phase and isc_arrival.pick_id.get_referred_object().waveform_id.station_code == station_code:
                                return isc_arrival
                return None

            anomaly_isc = lambda ids: find_related_isc_arrival(ids).time_residual if find_related_isc_arrival(ids) else np.NaN
            anomaly_rev = lambda ids: ( anomaly_isc(ids)
                                             + (ResourceIdentifier(ids[0]).get_referred_object().pick_id.get_referred_object().time
                                             - find_related_isc_arrival(ids).pick_id.get_referred_object().time) ) if find_related_isc_arrival(ids) else np.NaN

            df = pd.merge(df, pd.DataFrame([[
                        arrival_id,
                        round(anomaly_rev((arrival_id, event_id)), 2),
                        anomaly_isc((arrival_id, event_id))
                    ] for arrival_id, event_id in tqdm(zip(df['arrival_id'].values, df['event_id'].values), total=len(df['arrival_id'].values), desc="Processing arrivals")],
                columns=['arrival_id', 'anomaly_rev', 'anomaly_isc']
                ), on='arrival_id', how='inner')

        return df if include_id else df.drop(['arrival_id', 'event_id'], axis='columns')

if __name__ == "__main__":

    ### the whole loop may take a couple of hours, can be broken down parallel
    # for year in range(2010,2024):
    for year in range(2023,2024):
        for month in range(1):
        ### i/o configuration
            old_version = "1.2.2"
            new_version = "1.2.2"
            old_filename_suffix = f"_{year}"
            new_filename_subfix = f"_{year}"
            datalist_dir = "/Users/junsu/Documents/data_gcmt.pkl"
            event_filter = ""
            result_csv_filenames = glob.glob(f"result_stn_csv/updeANMO_shift5_pred_catalog_*{year}/*_outputs/X_prediction_results.csv")

            old_catalog_dir = f"result_cat/globocat_{old_version}{old_filename_suffix}.xml"
            new_catalog_dir = f"result_cat/globocat_{new_version}{new_filename_subfix}.xml"
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


        ### update catalog and save as new file
            globocat.resource_id=str(f"quakeml:jun.su/globocat_{new_version}")
            globocat.creation_info=CreationInfo(author="Jun Su", version=new_version, creation_time=UTCDateTime.now())

            with open(datalist_dir, 'rb') as f: datalist = pickle.load(f)
            globocat.update_catalog(datalist, result_csv_filenames, new_version, month=None)
            print('Saving file...', end="\r")
            globocat.write(new_catalog_dir, format='QUAKEML')
            print("Catalog is saved:", new_catalog_dir)
            
        ### export pandas dataframe
            df = globocat.get_dataframe(load_station_dict="station_dict.json", include_id=False, reference_isc=True)
            df.to_pickle(f'result_table_centroid/globocat_{new_version}{new_filename_subfix}_vs_isc.pkl')
