
import os
import sys
import glob
import json
import h5py
import numpy as np
import pandas as pd
import random
import pickle
# from multiprocessing.connection import Client
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth

# fn_starttime = lambda srctime: srctime - 0.5 * 60 * 60
# fn_endtime = lambda srctime: srctime + 2 * 60 * 60
fn_starttime = lambda srctime: srctime - 100
fn_endtime = lambda srctime: srctime + 1400

with open('sta2net.json') as sta2net_json:
    sta2net = json.load(sta2net_json)

class Event():
    def __init__(self, time: UTCDateTime, lat, lon, dep):
        self.srctime = time
        self.srcloc = (lat, lon, dep)
        self.stations = []
    def __eq__(self, target):
        return (self.srctime == target.srctime) & (self.srcloc == target.srcloc)
    def __ne__(self, target):
        return not self.__eq__(target)
    def __len__(self):
        return len(self.stations)
    def _findstation(self, target):
        for station in self.stations:
            if station == target: return station
        return None

class Station():
    def __init__(self, station, lat, lon, dist, azi):
        self.labelsta = {'name': station, 'lat': lat, 'lon': lon, 'dist': dist, 'azi': azi}
        self.labelnet = {'code': None}
        ## todo: find real sta
        self.records = []
        self.isdataexist = False
    def __eq__(self, target):
        return (self.labelsta['name'] == target.labelsta['name'])
    def __ne__(self, target):
        return not self.__eq__(target)

class Record():
    def __init__(self, phase, residual, error, ellipcor, crustcor, obstim, calctim):
        self.phase = phase
        self.residual = residual
        self.error = error
        self.ellipcor = ellipcor
        self.crustcor = crustcor
        self.obstim = obstim
        self.calctim = calctim

class SeismicData():
    def _getRefResponse(self):
        t0 = UTCDateTime(1989, 1, 1)
        return self.client.get_stations(network="SR", station="ANMO", channel="LH*", 
                            level="response", 
                            starttime=t0, endtime=t0+1)
    def _findevent(self, target):
        for event in self.events:
            if event == target: return event
        return None
    def _table2events(self, filename):
        table = pd.read_csv(filename, delim_whitespace=True)
        numNewEvents = 0
        # for index, row in table[0:100].iterrows(): #quick
        for index, row in table.iterrows(): #full
            timestamp = f"{row['year']}-{row['day']:03}T{row['hour']}:{row['min']}:{min(row['sec'],59.999)}Z"
            try: nlevent = Event(time=UTCDateTime(timestamp),
                    lat=row['eqlat'], lon=row['eqlon'], dep=row['eqdep'])
            except: print(row)
            nlstation = Station(row['station'], row['stalat'], row['stalon'], row['dist'], row['azi'])
            nlrecord = Record(row['phase'], row['residual'], row['error'], row['ellipcor'], row['crustcor'], row['obstim'], row['calctim'])
            matchevent = self._findevent(nlevent)
            if matchevent is None:
                # print(f"event not found, new event {timestamp} added")
                nlstation.records.append(nlrecord)
                nlevent.stations.append(nlstation)
                self.events.append(nlevent)
                numNewEvents += 1
            else:
                matchstation = matchevent._findstation(nlstation)
                if matchstation is None:
                    nlstation.records.append(nlrecord)
                    matchevent.stations.append(nlstation)
                else:
                    matchstation.records.append(nlrecord)
        return numNewEvents

    def fetch(self, skip_existing_events=True, skip_existing_stations=True):
        numdownloaded = 0
        for event in self.events:
            srctime = event.srctime
            srctime.precision = 3
            starttime = fn_starttime(srctime)
            endtime = fn_endtime(srctime)

            savedir = f"./training/{srctime}"
            print(savedir)
            
            # if (not (os.path.exists(savedir) and skip_existing_events)) and srctime>UTCDateTime("2009-09-11T00:00:00Z"):
            if (not (os.path.exists(savedir) and skip_existing_events)):
                # try:
                #     cat = client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC",
                #         minlatitude=event.srcloc[0]-0.1, maxlatitude=event.srcloc[0]+0.1,
                #         minlongitude=event.srcloc[1]-0.1, maxlongitude=event.srcloc[1]+0.1)
                # except:
                #     print(srctime, '... X, event not found')
                #     # cat = client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC")
                #     # if not len(cat) == 1:
                #     #     raise ValueError("None or more than 1 events found.")

                for station in event.stations:
                    ## revise the following line 
                    if not (glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy") and skip_existing_stations):
                        try:
                            inv = client.get_stations(station=f"{station.labelsta['name']}", starttime=starttime, endtime=endtime)

                        except:
                            try:
                                inv = client.get_stations(starttime=starttime, endtime=endtime,
                                        minlatitude=station.labelsta['lat']-0.02, maxlatitude=station.labelsta['lat']+0.02,
                                        minlongitude=station.labelsta['lon']-0.02, maxlongitude=station.labelsta['lon']+0.02)
                            except:
                                print(srctime, station.labelsta['name'], '... X (station not exists)')
                        # print(inv.networks[0].code)
                        for network in inv.networks:
                            if network.code == "SY":
                                inv.networks.remove(network)
                        
                        if len(inv.networks) > 0 :
                            try:
                                if not os.path.exists(savedir): os.makedirs(savedir)
                                st_raw = client.get_waveforms(inv.networks[0].code, inv.networks[0].stations[0].code, "*", "LH*", starttime, endtime, attach_response=True,
                                    filename=f"{savedir}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy")
                                print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})")
                                station.labelnet['code'] = inv.networks[0].code
                                station.isdataexist = True
                                numdownloaded += 1
                            except:
                                print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})", '... X (data not exists)')
                            # baz = gps2dist_azimuth(lat1=cat[0].origins[0].latitude, lon1=cat[0].origins[0].longitude, lat2=inv[0][0].latitude, lon2=inv[0][0].longitude)

                            # st = st_raw.copy()
                            # st[0:3] = st.remove_response(output="DISP", pre_filt=(0.005, 0.006, 30.0, 35.0)) \
                            #     .attach_response(inv)
        print(f"{numdownloaded} seismograms are downloaded.")
                
    def get_datalist(self, resample=0, rotate=True):
        ref_response = self._getRefResponse()
        with h5py.File('./test.hdf5','w') as f:
            datalist = []
            f.create_group("data")
            for event in data.events:
                for trace in event.stations:
                    # p info
                    p_arrival_sample = None
                    p_status = None
                    p_weight = None
                    p_travel_sec = None

                    for record in trace.records:
                        if record.phase == 'P':
                            p_status = 'manual'
                            p_weight = 1/record.error
                            # p_travel_sec = event.srctime + record.obstim - fn_starttime(event.srctime)
                            p_calctim = event.srctime + record.calctim
                            p_travel_sec = event.srctime + record.obstim - fn_starttime(p_calctim)
                            p_arrival_sample = int(p_travel_sec)
                            # p_arrival_sample = int(p_travel_sec - p_calctim)

                    # s info
                    s_arrival_sample = None
                    s_status = None
                    s_weight = None
                    s_travel_sec = None

                    for record in trace.records:
                        if record.phase == 'S':
                            s_status = 'manual'
                            s_weight = 1/record.error
                            # s_travel_sec = event.srctime + record.obstim - fn_starttime(event.srctime)
                            # s_arrival_sample = int(s_travel_sec)
                            s_obstim0 = event.srctime + record.obstim

                    obsfilenames = glob.glob(f"./training/{event.srctime}/*{trace.labelsta['name']}.LH.obspy")
                    if len(obsfilenames) > 0:
                        # network_code = None if (not trace.isdataexist) else sta2net[trace.labelsta['name']]['network']
                        # trace_name = None if (not trace.isdataexist) else f"{sta2net[trace.labelsta['name']]['network']}.{trace.labelsta['name']}.LH.obspy"
                        # network_code = sta2net[trace.labelsta['name']]['network']
                        # obsfile_name = f"{sta2net[trace.labelsta['name']]['network']}.{trace.labelsta['name']}.LH.obspy"
                        obsfile_name = obsfilenames[0]
                        network_code = obsfile_name.split('/')[-1].split('.')[0]
                        event_name = f"{trace.labelsta['name']}.network_code_{event.srctime.year:4d}{event.srctime.month:02d}{event.srctime.day:02d}{event.srctime.hour:02d}{event.srctime.minute:02d}{event.srctime.second:02d}_EV"

                        event_obspy = read(obsfile_name)
                        # rotate to RTZ coordinate
                        if rotate is True:
                            baz = gps2dist_azimuth(lat1=event.srcloc[0], lon1=event.srcloc[1], lat2=trace.labelsta['lat'], lon2=trace.labelsta['lon'])
                            event_obspy.rotate('NE->RT', back_azimuth=baz[2])

                        # resample
                        # if len(event_obspy) == 3 and len(event_obspy[0])*len(event_obspy[1])*len(event_obspy[2])>0 and all([np.isscalar(i) for i in event_obspy[0].data]) and all([np.isscalar(i) for i in event_obspy[1].data]) and all([np.isscalar(i) for i in event_obspy[2].data]):
                        if len(event_obspy) == 3 and len(event_obspy[0])*len(event_obspy[1])*len(event_obspy[2])>0 and np.isscalar(event_obspy[0].data[0]):
                            if resample != 0:
                                delta = 1/resample
                                event_obspy.trim(starttime=p_calctim-100, endtime=p_calctim+1400)
                                event_obspy.interpolate(sampling_rate=resample, method='lanczos', a=20)
                                if len(event_obspy) == 3:
                                    stdshape = (int(1500*resample), 3)
                                    event_data = np.transpose([np.array(event_obspy[0].data, dtype=np.float32), np.array(event_obspy[1].data, dtype=np.float32), np.array(event_obspy[2].data, dtype=np.float32)]) 
                                    if event_data.shape[0] <= int(1520*resample) and event_data.shape[0] > int(1500*resample): event_data = event_data[:int(1500*resample),0:3]
                            else:
                                delta = 1
                                # stdshape = (9000, 3)
                                # event_data = np.transpose([np.array(event_obspy[0].data, dtype=np.float64), np.array(event_obspy[1].data, dtype=np.float64), np.array(event_obspy[2].data, dtype=np.float64)]) 
                                # if event_data.shape[0] <= 9020 and event_data.shape[0] > 9000: event_data = event_data[:9000,0:3]
                                event_obspy.trim(starttime=p_calctim-100, endtime=p_calctim+1400)
                                stdshape = (1500, 3)
                                if len(event_obspy) == 3:
                                    event_data = np.transpose([np.array(event_obspy[0].data, dtype=np.float64), np.array(event_obspy[1].data, dtype=np.float64), np.array(event_obspy[2].data, dtype=np.float64)]) 
                                    if event_data.shape[0] <= 1520 and event_data.shape[0] > 1500: event_data = event_data[:1500,0:3]
                            # print(event_data.shape)

                            try:
                                event_data = event_data.astype(np.float32)
                                anynan = np.isnan(event_data).any()
                                conditions = (p_status and s_status and event_data.shape==stdshape and not anynan)

                                if conditions:
                                    s_arrival_sample = int(s_obstim0-fn_starttime(p_calctim))
                                    snr = np.sum(abs(event_data[int((s_arrival_sample-10)/delta):int((s_arrival_sample+50)/delta),:]), axis=0) / np.sum(abs(event_data[0:int(40/delta),:]), axis=0)
                                    dataset = f.create_dataset(f"data/{event_name}",data=event_data)
                                    dataset.attrs['p_arrival_sample'] = int(p_arrival_sample/delta)
                                    dataset.attrs['p_status'] = p_status
                                    dataset.attrs['p_weight'] = p_weight
                                    dataset.attrs['s_arrival_sample'] = int(s_arrival_sample/delta)
                                    dataset.attrs['s_status'] = s_status
                                    dataset.attrs['s_weight'] = s_weight
                                    dataset.attrs['coda_end_sample'] = int((s_arrival_sample-60)/delta)
                                    dataset.attrs['snr_db'] = snr
                                    dataset.attrs['trace_category'] = 'earthquake_local'
                                    dataset.attrs['network_code'] = network_code
                                    dataset.attrs['source_id'] = 'None'
                                    dataset.attrs['source_distance_km'] = trace.labelsta['dist'] * 111.1
                                    dataset.attrs['trace_name'] = event_name      
                                    dataset.attrs['trace_start_time'] = str(event.srctime)
                                    dataset.attrs['source_magnitude'] = 0
                                    dataset.attrs['receiver_type'] = 'LH'
                                    datalist.append({'network_code': network_code, 'receiver_code': trace.labelsta['name'], 'receiver_type': 'LH', 'receiver_latitude': trace.labelsta['lat'], 'receiver_longitude': trace.labelsta['lon'], 'receiver_elevation_m': None, 'p_arrival_sample': int(p_arrival_sample/delta), 'p_status': p_status, 'p_weight': p_weight, 'p_travel_sec': p_travel_sec, 's_arrival_sample': int(s_arrival_sample/delta), 's_status': s_status, 's_weight': s_weight, 'source_id': None, 'source_origin_time': event.srctime, 'source_origin_uncertainty_sec': None, 'source_latitude':event.srcloc[0], 'source_longitude': event.srcloc[1], 'source_error_sec': None, 'source_gap_deg': None, 'source_horizontal_uncertainty_km': None, 'source_depth_km': event.srcloc[2], 'source_depth_uncertainty_km': None, 'source_magnitude': None, 'source_magnitude_type': None, 'source_magnitude_author': None, 'source_mechanism_strike_dip_rake': None, 'source_distance_deg': trace.labelsta['dist'], 'source_distance_km': trace.labelsta['dist'] * 111.1, 'back_azimuth_deg': trace.labelsta['azi'], 'snr_db': snr, 'coda_end_sample': [[int((s_arrival_sample-60)/delta)]], 'trace_start_time': event.srctime, 'trace_category': 'earthquake_local', 'trace_name': event_name})
                            except:
                                print(f"Value error: {obsfile_name}")

        return datalist


    def __init__(self, client: Client, tables: list, autofetch=False):
        self.client = client
        self.refinv = self._getRefResponse()

        # create event list from tables
        self.events = []
        for table in tables:
            numevent = self._table2events(table)
            print(f"table read successfully, {numevent} events added")

        # fetch data if autofetch toggled
        if autofetch: self.fetch(skip_existing_events=False)
    

class Picker():
    def __init__(self):
        pass
    def train(self, data: SeismicData):
        pass

if __name__ == '__main__':
    client = Client('IRIS')

    # create dataset from scretch
    # path = "/Users/jun/Downloads/drive-download-20220512T014633Z-001"
    # data = SeismicData(client, [f"{path}/Pcomb.4.07.09.table",f"{path}/Scomb.4.07.09.table"], autofetch=True)
    # with open('data.pkl', 'wb') as outp:
    #     pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    # open existing dataset
    # with open('data.pkl', 'rb') as inp:
    #     loaddata = pickle.load(inp)
    # data = SeismicData(client, [])
    # data.events = loaddata.events

    # fetch
    # data.fetch(skip_existing_events=False)
    # with open('data_fetched.pkl', 'wb') as outp:
    #     pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    # just open existing fetched dataset
    with open('data_fetched.pkl', 'rb') as inp:
        loaddata = pickle.load(inp)
    data = SeismicData(client, [])
    data.events = loaddata.events
    
    # testing output
    printphase = [record.phase for record in data.events[0].stations[3].records]
    print(f"#1 event has {len(data.events[0])} stations, records of #4 event: {', '.join(printphase)}")

    # create dataframe
    # datalist = data.get_datalist(resample=8.0)
    datalist = data.get_datalist(resample=4.0)
    random.shuffle(datalist)
                
    df = pd.DataFrame(datalist[:int(0.7*len(datalist))])
    df.to_csv('training_PandS_up.csv', index=False)
    df = pd.DataFrame(datalist[int(0.7*len(datalist)):])
    df.to_csv('training_PandS_up_test.csv', index=False)
