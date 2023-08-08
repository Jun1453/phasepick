
import os
import sys
import glob
import json
import h5py
import time
import struct
import numpy as np
import pandas as pd
import random
import pickle
# from multiprocessing.connection import Client
from multiprocessing import Pool
from scipy.fft import fft, ifft
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.invsim import simulate_seismometer

fn_starttime_full = lambda srctime: srctime - 0.5 * 60 * 60
fn_endtime_full = lambda srctime: srctime + 2 * 60 * 60
fn_starttime_train = lambda srctime: srctime - 100
fn_endtime_train = lambda srctime: srctime + 1400

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

class InstrumentResponse():
    def __init__(self, network, station, component, start_end_times=None, timestamp=None):
        self.component = component
        self.sensitivity = {}

        # fill blank char
        if len(network) == 1: network += '-'
        if len(station) == 3: station += '-'

        if start_end_times:
            filename = f"/Users/jun/phasepick/resp.dir/{network}.{station}.{component}.{start_end_times}"
        elif timestamp:
            if type(timestamp) is not UTCDateTime: timestamp = UTCDateTime(timestamp)
            for search in glob.glob(f"/Users/jun/phasepick/resp.dir/{network}.{station}.{component}.*"):
                elements = search.split('/')[-1].split('.')
                resp_start = UTCDateTime(year=int(elements[-6]), julday=int(elements[-5]), hour=int(elements[-4]))
                resp_end = UTCDateTime(year=int(elements[-3]), julday=int(elements[-2]), hour=int(elements[-1]))
                if timestamp > resp_start and timestamp < resp_end:
                    filename = search
                    break
        else:
            raise ValueError("At least one parameter has to be given: start_end_times, timestamp")
        
        try:
            if filename:
                with open(filename, 'rb') as file:
                    buffer0, niris, freql, freqh, buffer1 = struct.unpack('>iiffi', file.read(20))
                    for k in range(niris):
                        buffer0, stfreq, streal, stimag, buffer1 = struct.unpack('>ifffi', file.read(20))
                        self.sensitivity[stfreq] = complex(-streal, -stimag)
        except UnboundLocalError:
            print(f'Response file not found: {start_end_times if start_end_times else timestamp} for {network}.{station}.{component}')
        except FileNotFoundError:
            print('Response file not found:', filename)

class SeismicData():
    # def _getRefResponse(self):
    #     # t0 = UTCDateTime(1989, 1, 1)
    #     # return self.client.get_stations(network="SR", station="ANMO", channel="LH*", 
    #     #                     level="response", 
    #     #                     starttime=t0, endtime=t0+1)
    #     t0 = UTCDateTime(1983, 1, 1)
    #     return self.client.get_stations(network="SR", station="GRFO", channel="LH*", 
    #                         level="response", 
    #                         starttime=t0, endtime=t0+1)
    def _findevent(self, target):
        for event in self.events:
            if event == target: return event
        return None
    def _table2events(self, filename):
        table = pd.read_csv(filename, delim_whitespace=True)
        numNewEvents = 0
        # for index, row in table[0:100].iterrows(): #quick
        for index, row in table.iterrows(): #full
            timestamp = f"{row['year']}-{row['day']:03}T{row['hour']:02}:{row['min']:02}:{min(row['sec'],59.999)}Z"
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
    
    def _fetch_par(self, event, skip_existing_events=False, skip_existing_stations=True):
        srctime = event.srctime
        srctime.precision = 3
        starttime = fn_starttime_full(srctime)
        endtime = fn_endtime_full(srctime)

        savedir = f"./rawdata/{srctime}"
        # print(savedir)
        
        # if (not (os.path.exists(savedir) and skip_existing_events)) and srctime>UTCDateTime("2009-09-11T00:00:00Z"):
        if (not (os.path.exists(savedir) and skip_existing_events)):
            # try:
            #     cat = self.client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC",
            #         minlatitude=event.srcloc[0]-0.1, maxlatitude=event.srcloc[0]+0.1,
            #         minlongitude=event.srcloc[1]-0.1, maxlongitude=event.srcloc[1]+0.1)
            # except:
            #     print(srctime, '... X, event not found')
            #     # cat = self.client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC")
            #     # if not len(cat) == 1:
            #     #     raise ValueError("None or more than 1 events found.")

            for station in event.stations:
                ## revise the following line 
                if not (glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy") and skip_existing_stations):
                    try:
                        inv = self.client.get_stations(station=f"{station.labelsta['name']}", starttime=starttime, endtime=endtime)

                    except:
                        try:
                            inv = self.client.get_stations(starttime=starttime, endtime=endtime,
                                    minlatitude=station.labelsta['lat']-0.02, maxlatitude=station.labelsta['lat']+0.02,
                                    minlongitude=station.labelsta['lon']-0.02, maxlongitude=station.labelsta['lon']+0.02)
                        except:
                            print(srctime, station.labelsta['name'], '... X (station not exists)')
                            continue
                    # print(inv.networks[0].code)
                    for network in inv.networks:
                        if network.code == "SY":
                            inv.networks.remove(network)
                    
                    if len(inv.networks) > 0 :
                        try:
                            if not os.path.exists(savedir): os.makedirs(savedir)
                            st_raw = self.client.get_waveforms(inv.networks[0].code, inv.networks[0].stations[0].code, "*", "LH*", starttime, endtime, attach_response=True,
                                filename=f"{savedir}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy")
                            print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})")
                            station.labelnet['code'] = inv.networks[0].code
                            station.isdataexist = True
                            self.numdownloaded += 1

                            # sleep or IRIS may cut down your connection
                            time.sleep(1)
                        except:
                            print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})", '... X (data not exists)')

    def fetch(self, skip_existing_events=False, skip_existing_stations=False):
        self.numdownloaded = 0
        p = Pool(10) # set parallel fetch
        p.map(self._fetch_par, self.events)
        # for event in self.events:
        #     srctime = event.srctime
        #     srctime.precision = 3
        #     starttime = fn_starttime_full(srctime)
        #     endtime = fn_endtime_full(srctime)

        #     savedir = f"./rawdata/{srctime}"
        #     print(savedir)
            
        #     # if (not (os.path.exists(savedir) and skip_existing_events)) and srctime>UTCDateTime("2009-09-11T00:00:00Z"):
        #     if (not (os.path.exists(savedir) and skip_existing_events)):
        #         # try:
        #         #     cat = self.client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC",
        #         #         minlatitude=event.srcloc[0]-0.1, maxlatitude=event.srcloc[0]+0.1,
        #         #         minlongitude=event.srcloc[1]-0.1, maxlongitude=event.srcloc[1]+0.1)
        #         # except:
        #         #     print(srctime, '... X, event not found')
        #         #     # cat = self.client.get_events(starttime=srctime-3, endtime=srctime+3, minmagnitude=5, catalog="ISC")
        #         #     # if not len(cat) == 1:
        #         #     #     raise ValueError("None or more than 1 events found.")

        #         for station in event.stations:
        #             ## revise the following line 
        #             if not (glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy") and skip_existing_stations):
        #                 try:
        #                     inv = self.client.get_stations(station=f"{station.labelsta['name']}", starttime=starttime, endtime=endtime)

        #                 except:
        #                     try:
        #                         inv = self.client.get_stations(starttime=starttime, endtime=endtime,
        #                                 minlatitude=station.labelsta['lat']-0.02, maxlatitude=station.labelsta['lat']+0.02,
        #                                 minlongitude=station.labelsta['lon']-0.02, maxlongitude=station.labelsta['lon']+0.02)
        #                     except:
        #                         print(srctime, station.labelsta['name'], '... X (station not exists)')
        #                 # print(inv.networks[0].code)
        #                 for network in inv.networks:
        #                     if network.code == "SY":
        #                         inv.networks.remove(network)
                        
        #                 if len(inv.networks) > 0 :
        #                     try:
        #                         if not os.path.exists(savedir): os.makedirs(savedir)
        #                         st_raw = self.client.get_waveforms(inv.networks[0].code, inv.networks[0].stations[0].code, "*", "LH*", starttime, endtime, attach_response=True,
        #                             filename=f"{savedir}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy")
        #                         print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})")
        #                         station.labelnet['code'] = inv.networks[0].code
        #                         station.isdataexist = True
        #                         numdownloaded += 1
        #                     except:
        #                         print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})", '... X (data not exists)')

        print(f"{self.numdownloaded} seismograms are downloaded.")

    def link_downloaded(self, israwdata=True):
        count = 0
        for event in self.events:
            srctime = event.srctime
            srctime.precision = 3
            savedir = f"./rawdata/{srctime}" if israwdata else f"./training/{srctime}" 
            for station in event.stations:
                search = glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy")
                if len(search) > 0:
                    obspyfile = read(search[0])
                    station.labelnet['code'] = obspyfile[0].meta.network
                    station.isdataexist = True
                    if israwdata: station.rawdatapath = search[0]
                    else: station.trainingpath = search[0]
                    count += 1
        print(f"{count} files are linked to event list.")

    def remove_response(self, rotate=False): # do not use
    #     for event in self.events:
    #         savedir = f"./training/{event.srctime}"
    #         if not os.path.exists(savedir): # to skip existing folders
    #             for trace in event.stations:
    #                 obsfilenames = glob.glob(f"./rawdata/{event.srctime}/*{trace.labelsta['name']}.LH.obspy")
    #                 if len(obsfilenames) > 0:
    #                     stream = read(obsfilenames[0])
    #                     if not os.path.exists(savedir): os.makedirs(savedir)
    #                     try: 
    #                         inv = self.client.get_stations(network=stream[0].stats.network, station=stream[0].stats.station, channel="LH*", 
    #                             level="response", 
    #                             starttime=stream[0].stats.starttime, endtime=stream[0].stats.starttime+1)
    #                         newstream = stream.copy()
    #                         # remove station response
    #                         newstream.remove_response(inventory=inv, pre_filt=(0.005, 0.006, 30.0, 35.0))
    #                         # add GRFO response
    #                         for i in range(3):
    #                             j = i if len(self.refinv[0][0])>1 else 0
    #                             sts2 = {'gain': self.refinv[0][0][j].response.get_paz().stage_gain if self.refinv[0][0][j].response.get_paz().stage_sequence_number>1 else 1, #gain set as 1 if only one stage sequence
    #                                 'poles': self.refinv[0][0][j].response.get_paz().poles,
    #                                 'sensitivity': self.refinv[0][0][j].response.instrument_sensitivity.value,
    #                                 'zeros': self.refinv[0][0][j].response.get_paz().zeros}
    #                             newstream[i].data = simulate_seismometer(newstream[i].data, samp_rate=newstream[0].stats.sampling_rate, paz_simulate=sts2)
    #                         # for i in range(3):
    #                         #     sts2 = {'gain': inv[0][0][i].response.get_paz().stage_gain,
    #                         #         'poles': inv[0][0][i].response.get_paz().poles,
    #                         #         'sensitivity': inv[0][0][i].response.instrument_sensitivity.value,
    #                         #         'zeros': inv[0][0][i].response.get_paz().zeros}
    #                         #     newstream[i].data = simulate_seismometer(newstream[i].data, samp_rate=newstream[0].stats.sampling_rate, paz_simulate=sts2)

    #                         # rotate coordinate from NEZ to RTZ
    #                         if rotate: 
    #                             # baz = gps2dist_azimuth(lat1=inv[0][0].latitude, lon1=inv[0][0].longitude, lat2=self.refinv[0][0].latitude, lon2=self.refinv[0][0].longitude)
    #                             baz = gps2dist_azimuth(lat1=event.srcloc[0], lon1=event.srcloc[1], lat2=trace.labelsta['lat'], lon2=trace.labelsta['lon'])
    #                             newstream.rotate('NE->RT', back_azimuth=baz[2])

    #                         # save data
    #                         newstream.write(f"{savedir}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy", format="PICKLE")

    #                     except:
    #                         print(f"cannot remove response for {event.srctime}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy")
            pass

    def deconvolve(self, rawdata, station_components, reference_components):
        proceed = rawdata.copy()

        # loop for all (three) components
        for i in range(len(station_components)):

            # match waveform compenent with reference response
            station = station_components[i]
            for reference in reference_components:
                if reference.component == station.component: break
            if reference.component != station.component: raise ValueError(f"Cannot match component for waveform and reference response: {str(rawdata)}")

            # fast-fourier-transform the trace
            fdomain_data = fft(rawdata[i].data)

            # get frequency array from loaded response
            freq = np.array(list(reference.sensitivity.keys()))

            # interpolate frequency array to fit waveform data
            freq_interp = np.linspace(freq.min(), freq.max(), num=len(fdomain_data)//2)

            # interpolate reference and target station response
            ref_resp_interp = np.interp(freq_interp, list(reference.sensitivity.keys()), list(reference.sensitivity.values()))
            sta_resp_interp = np.interp(freq_interp, list(station.sensitivity.keys()), list(station.sensitivity.values()))

            # deconvolve and convolve the trace
            fdomain_data[:len(fdomain_data)//2] = fdomain_data[:len(fdomain_data)//2] / sta_resp_interp * ref_resp_interp
            fdomain_data[len(fdomain_data)//2:] = np.flip(fdomain_data[:len(fdomain_data)//2]) / sta_resp_interp * ref_resp_interp
            fdomain_data[0] = complex(0, 0)

            # inverse fast-fourier-transform the trace
            deconvolved_data = ifft(fdomain_data)
            proceed[i].data = deconvolved_data / max(deconvolved_data)

        return proceed

    def get_datalist(self, resample=0, rotate=True, preprocess=True, output='./test.hdf5'):
        
        # get instrument response for reference station
        # ref_response = self._getRefResponse() # do not use
        reference_responses = [InstrumentResponse(network='SR', station='GRFO', component=component, timestamp=UTCDateTime(1983, 1, 1)) for component in ['LHE', 'LHN', 'LHZ']]
        # reference_responses = [InstrumentResponse(network='SR', station='GRFO', component='LHZ', start_end_times='1978.225.00.1993.329.00')] * 3

        # set directory
        loaddir = 'rawdata' if preprocess else 'training'

        # open new hdf5 database
        with h5py.File(output,'w') as f:

            # create datalist by searching all record lists
            datalist = []
            f.create_group("data")

            for event in self.events:
                for trace in event.stations:
                    # random shifting
                    # rand.

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
                            p_travel_sec = event.srctime + record.obstim - fn_starttime_train(p_calctim)
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

                    obsfilenames = glob.glob(f"./{loaddir}/{event.srctime}/*{trace.labelsta['name']}.LH.obspy")
                    if len(obsfilenames) > 0 and event.srctime > UTCDateTime(1971, 1, 1):
                        # network_code = None if (not trace.isdataexist) else sta2net[trace.labelsta['name']]['network']
                        # trace_name = None if (not trace.isdataexist) else f"{sta2net[trace.labelsta['name']]['network']}.{trace.labelsta['name']}.LH.obspy"
                        # network_code = sta2net[trace.labelsta['name']]['network']
                        # obsfile_name = f"{sta2net[trace.labelsta['name']]['network']}.{trace.labelsta['name']}.LH.obspy"
                        obsfile_name = obsfilenames[0]
                        network_code = obsfile_name.split('/')[-1].split('.')[0]
                        event_name = f"{trace.labelsta['name']}.{network_code}_{event.srctime.year:4d}{event.srctime.month:02d}{event.srctime.day:02d}{event.srctime.hour:02d}{event.srctime.minute:02d}{event.srctime.second:02d}_EV"

                        stream = read(obsfile_name)

                        # sanity check for all three component
                        # if len(stream) == 3 and len(stream[0])*len(stream[1])*len(stream[2])>0 and all([np.isscalar(i) for i in stream[0].data]) and all([np.isscalar(i) for i in stream[1].data]) and all([np.isscalar(i) for i in stream[2].data]):
                        if len(stream) == 3 and len(stream[0])*len(stream[1])*len(stream[2])>0 and np.isscalar(stream[0].data[0]):
                            
                            # check array size for waveform data
                            wrongsize = False
                            for record in stream:
                                if record.data.shape[0] <= 9000+20 and record.data.shape[0] > 9000: record.data = record.data[:9000]
                                elif record.data.shape[0] > 9000+20: print(f"Trace has too many samples: {str(record)}"); wrongsize = True; break
                                elif record.data.shape[0] < 9000: print(f"Trace has too few samples: {str(record)}"); wrongsize = True; break
                            if wrongsize: continue
                            # print(record)
                            
                            try:
                                # preprocessing
                                if preprocess:
                                    # get instrument response for waveform station
                                    # station_response = InstrumentResponse(network='GE', station='UGM', component='LHZ', start_end_times='2006.180.00.3001.001.00')
                                    station_responses = [InstrumentResponse(network=network_code, station=trace.labelsta['name'], component=component, timestamp=event.srctime) for component in ['LHE', 'LHN', 'LHZ']]
                                    # station_responses = [InstrumentResponse(network='GE', station='UGM', component='LHZ', start_end_times='2006.180.00.3001.001.00')] * 3
                                    
                                    # deconvolve and convolve instrument response
                                    stream = self.deconvolve(rawdata=stream, station_components=station_responses, reference_components=reference_responses)
                                    
                                    # calculate azimuth angle
                                    azimuth = gps2dist_azimuth(lat1=trace.labelsta['lat'], lon1=trace.labelsta['lon'], lat2=event.srcloc[0], lon2=event.srcloc[1])[2] if rotate is True else 180
                                    # azimuth = gps2dist_azimuth(lat1=-7.913, lon1=110.523, lat2=-0.660, lon2=133.430)[2]
                                    
                                    # rotate to TRZ coordinate
                                    stream.rotate('NE->RT', back_azimuth=azimuth)

                                    # save data
                                    srctime = event.srctime
                                    srctime.precision = 3

                                    savedir = f"./training/{srctime}"
                                    if not os.path.exists(savedir): os.makedirs(savedir)
                                    stream.write(f"{savedir}/{network_code}.{trace.labelsta['name']}.LH.obspy", format="PICKLE")

                                # resampling
                                if resample != 0:
                                    delta = 1/resample
                                    stream.trim(starttime=p_calctim-100, endtime=p_calctim+1400)
                                    stream.interpolate(sampling_rate=resample, method='lanczos', a=20)
                                    if len(stream) == 3:
                                        stdshape = (int(1500*resample), 3)
                                        waveform_data = np.transpose([np.array(stream[0].data, dtype=np.float32), np.array(stream[1].data, dtype=np.float32), np.array(stream[2].data, dtype=np.float32)]) 
                                        if waveform_data.shape[0] <= int(1520*resample) and waveform_data.shape[0] > int(1500*resample): waveform_data = waveform_data[:int(1500*resample),0:3]
                                else:
                                    delta = 1
                                    # stdshape = (9000, 3)
                                    # waveform_data = np.transpose([np.array(stream[0].data, dtype=np.float64), np.array(stream[1].data, dtype=np.float64), np.array(stream[2].data, dtype=np.float64)]) 
                                    # if waveform_data.shape[0] <= 9020 and waveform_data.shape[0] > 9000: waveform_data = waveform_data[:9000,0:3]
                                    stream.trim(starttime=p_calctim-100, endtime=p_calctim+1400)
                                    stdshape = (1500, 3)
                                    if len(stream) == 3:
                                        waveform_data = np.transpose([np.array(stream[0].data, dtype=np.float64), np.array(stream[1].data, dtype=np.float64), np.array(stream[2].data, dtype=np.float64)]) 
                                        if waveform_data.shape[0] <= 1520 and waveform_data.shape[0] > 1500: waveform_data = waveform_data[:1500,0:3]

                                # check for problematic values in array
                                waveform_data = waveform_data.astype(np.float32)
                                anynan = np.isnan(waveform_data).any()
                                conditions = (p_status and s_status and waveform_data.shape==stdshape and not anynan)

                                if conditions:
                                    s_arrival_sample = int(s_obstim0-fn_starttime_train(p_calctim))
                                    snr = np.sum(abs(waveform_data[int((s_arrival_sample-10)/delta):int((s_arrival_sample+50)/delta),:]), axis=0) / np.sum(abs(waveform_data[0:int(40/delta),:]), axis=0)
                                    dataset = f.create_dataset(f"data/{event_name}",data=waveform_data)
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
                            except ValueError as e:
                                print(f"Value error for {obsfile_name}: {e}")
                            # except:
                            #     pass
                            # except:
                            #     print(f"Unexpect error for {obsfile_name}")

        return datalist


    def __init__(self, client: Client, tables: list, autofetch=False):
        self.client = client
        self.numdownloaded = 0
        # self.refinv = self._getRefResponse()

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
    # data = SeismicData(client, [f"{path}/Pcomb.4.07.09.table",f"{path}/Scomb.4.07.09.table"], autofetch=False)
    # with open('data.pkl', 'wb') as outp:
    #     pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    # open existing dataset
    # with open('data.pkl', 'rb') as inp:
    #     loaddata = pickle.load(inp)
    # data = SeismicData(client, [])
    # data.events = loaddata.events

    # fetch
    # time.sleep(600)
    # data.fetch()
    # # data.fetch(skip_existing_events=False)
    # with open('data_fetched.pkl', 'wb') as outp:
    #     pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    # link
    # data.link_downloaded(israwdata=True)
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

    # # # remove instrument response
    # # # data.remove_response(rotate=True)

    # create dataframe
    # datalist = data.get_datalist(resample=8.0)
    datalist = data.get_datalist(resample=4.0, rotate=True, preprocess=True, output='./updeANMO.hdf5')
    random.shuffle(datalist)
                
    # df = pd.DataFrame(datalist[:int(0.7*len(datalist))])
    df = pd.DataFrame(datalist)
    df.to_csv('training_PandS_updeANMO.csv', index=False)
    # df = pd.DataFrame(datalist[int(0.7*len(datalist)):])
    # df.to_csv('training_PandS_upANMO_test.csv', index=False)
