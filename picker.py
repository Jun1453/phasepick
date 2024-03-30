import io
import os
import csv
import glob
import json
import h5py
import time
import struct
import shutil
import numpy as np
import pandas as pd
import random
import pickle
import warnings
# import EQTransformer as eqt
# from functools import partial
from itertools import repeat
from multiprocessing import Pool, Manager, cpu_count
from multiprocessing.pool import ThreadPool
from scipy.fft import rfft, irfft
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read, read_inventory
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
from obspy.signal.invsim import simulate_seismometer, evalresp_for_frequencies
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning

warnings.simplefilter('ignore', category=ObsPyDeprecationWarning)

fn_starttime_full = lambda srctime: srctime - 0.5 * 60 * 60
fn_endtime_full = lambda srctime: srctime + 2 * 60 * 60
fn_starttime_train = lambda srctime: srctime - 250
fn_endtime_train = lambda srctime: srctime + 1250

# with open('sta2net.json') as sta2net_json:
#     sta2net = json.load(sta2net_json)

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
    def appendstation(self, station):
        self.stations.append(station)

class Station():
    def __init__(self, station, lat, lon, dist, azi, loc=''):
        self.labelsta = {'name': station, 'lat': lat, 'lon': lon, 'dist': dist, 'azi': azi, 'loc': loc}
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

class Instrument():
    # def _get_resp_path(self):
    #     pass

    # def _get_resp(self):
    #     if self.resp_path is None: self._get_resp_path()
    #     with open(self.resp_path, 'r') as f:
    #         self.resp = f.read()

    # def _evalresp(self, t_samp=None, fmin=0, fmax=None, nfreq=500, network=None, station=None, channel='*', locid='*', units='VEL', debug=False):

    #     if self.sampling_rate is None: self._get_sampling_rate()
    #     if self.resp_path is None: self._get_resp_path()
    #     if t_samp is None: t_samp = self.sampling_rate
    #     if fmax is None: fmax = 1 / (t_samp * 2.0)
    #     if network is None: network = self.network_code
    #     if station is None: station = self.station_code

    #     datetime = self.starttime+1
    #     freqs = np.linspace(fmin, fmax, nfreq)
    #     h = evalresp_for_frequencies(t_samp, freqs, self.resp_path, datetime, station, channel, network, locid, units, debug)
    #     return {freqs[i]: -h[i] for i in range(len(h))}
    
    # def resp_for_deconv(self, nfreq):
    #     if not nfreq in self.evaluated_response:
    #         self.evaluated_response[nfreq] = self._evalresp(fmin=1e-10, nfreq=nfreq)
    #     return self.evaluated_response[nfreq]
    
    def find_responses(self, datetime=None):
        """This function returns a dictionary of response objects of the given components that cover the given timestamp."""
        if type(datetime) is str: datetime = UTCDateTime(datetime)

        # get timestamp from first response of each set in the response inventory
        for component_set in self.resp_inventory:
            # return responses if found in resp_inventory
            if datetime > component_set.start_date and (not component_set.end_date or datetime < component_set.end_date):
                return component_set

        # download responses if not found in resp_inventory
        try:
            inv = self.client.get_stations(network=self.network_code, station=self.station_code, starttime=datetime, level='RESP')
            startdate = inv[0][0].start_date
            enddate = inv[0][0].end_date or UTCDateTime("30010101")
            inv.write(f"{self.respdir}/{self.network_code}.{self.station_code}.{startdate.year}.{startdate.julday:03}.{startdate.hour:02}.{enddate.year}.{enddate.julday:03}.{enddate.hour:02}", format="STATIONXML")
            self.resp_inventory.append(inv[0][0])
            return inv[0][0]
        except:
            print(f"No data: {self.network_code}.{self.station_code} {datetime}")
            return None

        # resp_dict = {}
        # for component in components:
            # resp_dict[component] = 
            # filelike = io.BytesIO(self.client.resp(self.network_code, self.station_code, "*", "LHZ,LHN,LHE", time=datetime))
            # filenames = glob.glob(f"{self.respdir}/{self.network_code}.{self.station_code}.{component}.*")
            # if len(filenames)==0: raise FileNotFoundError(f"response file not found for {network}.{station}.{component}")
            # if type(timestamp) is not UTCDateTime: timestamp = UTCDateTime(timestamp)
                

        # return resp_dict

    def __init__(self, station_str=str, event_records=list, respdir=str):
        self.respdir = respdir
        if not os.path.exists(self.respdir): os.mkdir(self.respdir)
        #print(station_str)
        self.network_code, self.station_code, _ = station_str.split('.')
        self.client = Client("IRIS")
        self.resp4record = {}
        self.resp_inventory = [read_inventory(filename)[0][0] for filename in glob.glob(f"{self.respdir}/{self.network_code}.{self.station_code}.*")]

        # self.sampling_rate = None
        # self.starttime = None
        # self.endtime = None
        # self.resp = list(Response)
        # resp_key = f"{self.network_code}.{self.station_code}.{channel}.{starttime.year}.{starttime.julday}.{starttime.hour}.{endtime.year}.{endtime.julday}.{endtime.hour}" #SR.GRFO.LHZ.1978.225.00.1993.329.00
        # self.resp[resp_key] = {
        #     'resp_path': str,
        #     'sampling_rate': float,
        # }
        # self.resp_path = None
        # self.evaluated_response = {int: dict}

        # fill blank char
        # if len(network_code) == 1: network_code += '-'
        # if len(self.station_code) == 3: self.station_code += '-'

        for datetime_str in event_records:
            self.resp4record[datetime_str] = self.find_responses(datetime_str)

    def __repr__(self):
        return f"{self.network_code}.{self.station_code}: "+', '.join([f"{sta.start_time.year}.{sta.start_time.julday}.{sta.start_time.hour}.{sta.end_time.year}.{sta.end_time.julday}.{sta.end_time.hour}" for sta in self.resp_inventory])
        

class InstrumentResponse():
    def __init__(self, network, station, component, start_end_times=None, timestamp=None):
        self.component = component
        self.sensitivity = {}
        filename = None

        # fill blank char
        if len(network) == 1: network += '-'
        if len(station) == 3: station += '-'

        if start_end_times:
            filename = f"/Users/jun/phasepick/resp.dir/{network}.{station}.{component}.{start_end_times}"
        elif timestamp:
            filenames = glob.glob(f"/Users/jun/phasepick/resp.dir/{network}.{station}.{component}.*")
            if len(filenames)==0: raise FileNotFoundError(f"response file not found for {network}.{station}.{component}")
            if type(timestamp) is not UTCDateTime: timestamp = UTCDateTime(timestamp)
            for search in filenames:
                elements = search.split('/')[-1].split('.')
                resp_start = UTCDateTime(year=int(elements[-6]), julday=int(elements[-5]), hour=int(elements[-4]))
                resp_end = UTCDateTime(year=int(elements[-3]), julday=int(elements[-2]), hour=int(elements[-1]))
                if timestamp > resp_start and timestamp < resp_end:
                    filename = search
                    break
        else:
            raise ValueError("at least one parameter has to be given: start_end_times, timestamp")
        
        if not filename:
            raise FileNotFoundError(f"no matched response file for {network}.{station}.{component}")
        
        with open(filename, 'rb') as file:
            buffer0, niris, freql, freqh, buffer1 = struct.unpack('>iiffi', file.read(20))
            for k in range(niris):
                buffer0, stfreq, streal, stimag, buffer1 = struct.unpack('>ifffi', file.read(20))
                self.sensitivity[stfreq] = complex(-streal, -stimag)
        # except UnboundLocalError:
        #     print(f'Response file not found: {start_end_times if start_end_times else timestamp} for {network}.{station}.{component}')
        # except FileNotFoundError:
        #     print('Response file not found:', filename)

class SeismicData():
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
    
    def _folder2events(self, filename):
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
    
    def _fetch_par(self, event, skip_existing_events=True, skip_existing_stations=True):
        srctime = event.srctime
        srctime.precision = 3
        starttime = fn_starttime_full(srctime)
        endtime = fn_endtime_full(srctime)

        # savedir = f"./rawdata/{srctime}"
        
        # if (not (os.path.exists(savedir) and skip_existing_events)):

        # find stations one by one according to the table
        if len(event.stations) > 0:
            savedir = f"./rawdata/{srctime}"
            if (not (os.path.exists(savedir) and skip_existing_events)):
                for station in event.stations:
                    ## revise the following line 
                    if not (glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy") and skip_existing_stations):
                        try:
                            inv = self.client.get_stations(station=f"{station.labelsta['name']}", starttime=starttime, endtime=endtime)

                        except:
                            try:
                                inv = self.client.get_stations(starttime=starttime, endtime=endtime, channel="LHZ,LHN,LHE",
                                        minlatitude=station.labelsta['lat']-0.02, maxlatitude=station.labelsta['lat']+0.02,
                                        minlongitude=station.labelsta['lon']-0.02, maxlongitude=station.labelsta['lon']+0.02)
                            except:
                                print(srctime, station.labelsta['name'], '... X (station not exists)')
                                continue
                        # print(inv.networks[0].code)
                        for network in inv.networks:
                            if network.code == "SY":
                                inv.networks.remove(network); continue
                            for station in network.stations:
                                if len(station.channels) != 3:
                                    network.stations.remove(station)
                            if len(network.stations) == 0:
                                inv.networks.remove(network)
                        
                        if len(inv.networks) > 0 :
                            try:
                                if not os.path.exists(savedir): os.makedirs(savedir)
                                st_raw = self.client.get_waveforms(inv.networks[0].code, inv.networks[0].stations[0].code, "*", "LHZ,LHN,LHE", starttime, endtime, attach_response=True,
                                    filename=f"{savedir}/{inv.networks[0].code}.{inv.networks[0].stations[0].code}.LH.obspy")
                                print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})")
                                station.labelnet['code'] = inv.networks[0].code
                                station.isdataexist = True

                                # sleep or IRIS may cut down your connection
                                time.sleep(1)
                            except:
                                print(srctime, f"{station.labelsta['name']} ({inv.networks[0].code}-{inv.networks[0].stations[0].code})", '... X (data not exists)')
            return event

        # when no stations are given, search for every availible
        else:
            # savedir = f"./rawdata_catalog/{srctime}"
            filename = f"./rawdata_catalog2/{srctime}.LH.obspy"
            # fetch data if no file exist
            if (not (os.path.exists(filename) and skip_existing_events)):
                return event
                try:
                    inv = self.client.get_stations(starttime=starttime, endtime=endtime, channel="LHZ,LHN,LHE")
                except:
                    print(srctime, '... X (no available station exists)')
                    return

                for network in inv.networks:
                    if network.code == "SY":
                        inv.networks.remove(network)
                    for station in network.stations:
                        if len(station.channels) != 3:
                            network.stations.remove(station)
                    if len(network.stations) == 0:
                        inv.networks.remove(network)
                
                if len(inv.networks) > 0 :
                    try:
                        self.client.get_waveforms("*", "*", "*", "LHZ,LHN,LHE", starttime, endtime, attach_response=True, filename=filename)
                    except:
                        print(srctime,  '... X (error in retrieving traces)')
                        return
                    
                    station_list = []
                    fetched_stream = read(filename)
                    for trace in fetched_stream:
                        try:
                            if not trace.stats.station in station_list:
                                traces_selected = fetched_stream.select(station=trace.stats.station)
                                network = inv.select(station=trace.stats.station)[0]
                                station = network[0]

                                # ignore trace if the only component
                                if len(traces_selected) < 3:
                                    continue
                                # record location if all traces agree
                                elif len(traces_selected) == 3:
                                    if (traces_selected[0].stats.location==traces_selected[1].stats.location) and (traces_selected[1].stats.location==traces_selected[2].stats.location):
                                        location = traces_selected[0].stats.location
                                    else: continue
                                elif len(traces_selected) > 3:
                                    # reselect for location '' if more than 3 components
                                    traces_selected = fetched_stream.select(station=trace.stats.station, location='')
                                    if len(traces_selected) == 3: location = traces_selected[0].stats.location
                                    # reselect for location '00' if more than 3 components
                                    else:
                                        traces_selected = fetched_stream.select(station=trace.stats.station, location='00')
                                        if len(traces_selected) == 3: location = traces_selected[0].stats.location
                                        else: continue
                                
                                fetched = Station(station.code, station.latitude, station.longitude,
                                                    dist=locations2degrees(lat1=station.latitude, long1=station.longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                                                    azi=gps2dist_azimuth(lat1=station.latitude, lon1=station.longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2],
                                                    loc=location)
                                fetched.labelnet['code'] = network.code
                                fetched.isdataexist = True
                                event.stations.append(fetched)
                                station_list.append(trace.stats.station)
                                
                        except:
                            print(srctime, trace.stats.station, '... X (error in reading fetched traces)')
                            station_list.append(trace.stats.station)
                    
                    print(srctime, f"Fetched for {len(event.stations)} stations")

                    # sleep or IRIS may cut down your connection
                    time.sleep(1)
                        
                    # if not os.path.exists(savedir): os.makedirs(savedir)
                    # for network in inv.networks:
                    #     for station in network.stations:
                    #         try:
                    #             st_raw = self.client.get_waveforms(network.code, station.code, "*", "LHZ,LHN,LHE", starttime, endtime, attach_response=True,
                    #                 filename=f"{savedir}/{network.code}.{station.code}.LH.obspy")
                    #             fetched = Station(station.code, station.latitude, station.longitude,
                    #                                 dist=locations2degrees(lat1=station.latitude, long1=station.longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                    #                                 azi=gps2dist_azimuth(lat1=station.latitude, lon1=station.longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2])
                    #             fetched.labelnet['code'] = network.code
                    #             fetched.isdataexist = True
                    #             event.stations.append(fetched)
                    #             print(srctime, f"Fetched ({network.code}-{station.code})")

                    #             # sleep or IRIS may cut down your connection
                    #             time.sleep(1)
                    #         except FDSNNoDataException as e:
                    #             print(srctime, f"{station.code} ({network.code}-{station.code})", '... X (data not exists)')
                    #         except: 
                    #             print(srctime, f"{station.code} ({network.code}-{station.code})", '... X (unknown issue)')
                else:
                    print(srctime, '... X (no available station exists)')
            # load data and get station metadata
            else:
                # return event
                # fetched_files = glob.glob(f"{savedir}/*.{station.labelsta['name']}.LH.obspy")
                loaded_stream = read(filename)
                station_list = []
                for trace in loaded_stream:
                    if not trace.stats.station in station_list:
                        try:
                            inv = self.client.get_stations(station=trace.stats.station, starttime=trace.stats.starttime, endtime=trace.stats.endtime, level="station")
                            fetched = Station(inv[0][0].code, inv[0][0].latitude, inv[0][0].longitude,
                                            dist=locations2degrees(lat1=inv[0][0].latitude, long1=inv[0][0].longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                                            azi=gps2dist_azimuth(lat1=inv[0][0].latitude, lon1=inv[0][0].longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2])
                            fetched.labelnet['code'] = inv[0].code
                            fetched.isdataexist = True
                            event.stations.append(fetched)
                        except:
                            pass
                            # print(srctime, trace.stats.station, '... X (failed loading)')
                        station_list.append(trace.stats.station)
                print(srctime, f"Loaded fetched traces for {len(event.stations)} stations")

            return event


    def fetch(self, cpu_number=12):
        self.numdownloaded = 0
        p = Pool(cpu_number) # set parallel fetch
        # self.events = p.map(self._fetch_par, self.events[:2000])
        self.events = p.map(self._fetch_par, self.events[32020:33872]) #only year 2010
        print("fetch finished.")
        for ev in self.events:
            if ev: self.numdownloaded+=len(ev.stations)
            else: self.events.remove(ev)
        print(f"{self.numdownloaded} seismograms are processed.")

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
            fdomain_data = rfft(rawdata[i].data)

            # get frequency array from loaded response
            freq = np.array(list(reference.sensitivity.keys()))

            # interpolate frequency array to fit waveform data
            freq_interp = np.linspace(freq.min(), freq.max(), num=len(fdomain_data))

            # interpolate reference and target station response
            ref_resp_interp = np.interp(freq_interp, list(reference.sensitivity.keys()), list(reference.sensitivity.values()))
            sta_resp_interp = np.interp(freq_interp, list(station.sensitivity.keys()), list(station.sensitivity.values()))

            # deconvolve and convolve the trace
            fdomain_data = fdomain_data / sta_resp_interp * ref_resp_interp
            fdomain_data[0] = complex(0, 0)

            # inverse fast-fourier-transform the trace
            deconvolved_data = irfft(fdomain_data)
            proceed[i].data = deconvolved_data / max(deconvolved_data)

        return proceed
    
    def _par_datalist(self, trace, args):
        event = args['event']
        obsfile = args['obsfile']
        preprocess = args['preprocess']
        resample = args['resample']
        shift = args['shift']
        rotate = args['rotate']
        reference_responses = args['reference_responses']
        loaddir = args['loaddir']
        obsfilenames = args['obsfilenames']
        savedir = args['savedir']

        # check basic arguments
        if obsfile == 'compiled':
            if trace.labelnet['code'] == 'SY': return
        elif obsfile == 'separate':
            obsfilenames = glob.glob(f"./{loaddir}/{event.srctime}/*{trace.labelsta['name']}.LH.obspy")
        else:
            raise Exception("wrong argument for `obsfile`")
        if not obsfilenames: raise Exception("no matched files for `obsfilenames`")

        # loading obspy file
        if len(obsfilenames) > 0 and event.srctime > UTCDateTime(1971, 1, 1):
            if obsfile == 'separate':
                obsfile_name = obsfilenames[0]
                stream = read(obsfile_name)
                network_code = obsfile_name.split('/')[-1].split('.')[0]

                # get instrument response for waveform station
                if preprocess:
                    try:
                        station_responses = [InstrumentResponse(network=network_code, station=trace.labelsta['name'], component=component, timestamp=event.srctime) for component in ['LHE', 'LHN', 'LHZ']]
                        # break if any response is missed
                        if len(station_responses[0].sensitivity)*len(station_responses[1].sensitivity)*len(station_responses[2].sensitivity) == 0: raise ValueError(f"response is missing for {network_code}.{trace.labelsta['name']}")
                    except ValueError as e:
                        print(f"Value error for {obsfile_name}: {e}"); return
                    except FileNotFoundError as e:
                        print(f"File-not-found error for {obsfile_name}: {e}"); return
            else:
                # get instrument response for waveform station
                network_code = trace.labelnet['code']
                obsfile_name = obsfilenames[0]
                if preprocess:
                    try:
                        station_responses = [InstrumentResponse(network=network_code, station=trace.labelsta['name'], component=component, timestamp=event.srctime) for component in ['LHE', 'LHN', 'LHZ']]
                        # break if any response is missed
                        if len(station_responses[0].sensitivity)*len(station_responses[1].sensitivity)*len(station_responses[2].sensitivity) == 0: raise ValueError(f"response is missing for {network_code}.{trace.labelsta['name']}")       
                    except ValueError as e:
                        print(f"Value error for {obsfile_name}: {e}"); return
                    except FileNotFoundError as e:
                        print(f"File-not-found error for {obsfile_name}: {e}"); return
                    
                stream = read(obsfile_name)
                stream = stream.select(station=trace.labelsta['name'], location=trace.labelsta['loc'])
            event_name = f"{trace.labelsta['name']}.{network_code}_{event.srctime.year:4d}{event.srctime.month:02d}{event.srctime.day:02d}{event.srctime.hour:02d}{event.srctime.minute:02d}{event.srctime.second:02d}_EV"

            # sanity check for all three component
            if len(stream) == 3 and len(stream[0])*len(stream[1])*len(stream[2])>0 and np.isscalar(stream[0].data[0]):
                
                try:
                    # check array size for waveform data
                    for record in stream:
                        if record.data.shape[0] <= 9000+20 and record.data.shape[0] > 9000: record.data = record.data[:9000]
                        elif record.data.shape[0] > 9000+20: raise ValueError(f"Trace has too many samples: {str(record)}")
                        elif record.data.shape[0] < 9000: raise ValueError(f"Trace has too few samples: {str(record)}")
                        
                    # preprocessing
                    print(f"preprocessing {event_name}...")
                    
                    if preprocess=='bandpass':
                        # remove instrument response
                        # inv = self.client.get_stations(station=f"{trace.labelsta['name']}", starttime=event.srctime, endtime=event.srctime+1)
                        # stream.remove_response(inventory=inv, pre_filt=(0.005, 0.006, 30.0, 35.0))
                        # calculate azimuth angle
                        azimuth = gps2dist_azimuth(lat1=trace.labelsta['lat'], lon1=trace.labelsta['lon'], lat2=event.srcloc[0], lon2=event.srcloc[1])[2] if rotate is True else 180
                        # rotate to TRZ coordinate
                        stream.rotate('NE->RT', back_azimuth=azimuth)
                        # bandpass filter
                        stream.filter('bandpass', freqmin=0.03, freqmax=0.05, corners=2, zerophase=True)

                    elif preprocess=='onlyrot':
                        # calculate azimuth angle
                        azimuth = gps2dist_azimuth(lat1=trace.labelsta['lat'], lon1=trace.labelsta['lon'], lat2=event.srcloc[0], lon2=event.srcloc[1])[2] if rotate is True else 180
                        # rotate to TRZ coordinate
                        stream.rotate('NE->RT', back_azimuth=azimuth)
                        # bandpass filter
                        # stream.filter('bandpass', freqmin=0.03, freqmax=0.05, corners=2, zerophase=True)

                    elif preprocess:
                        # deconvolve and convolve instrument response
                        stream = self.deconvolve(rawdata=stream, station_components=station_responses, reference_components=reference_responses)
                        
                        # calculate azimuth angle
                        azimuth = gps2dist_azimuth(lat1=trace.labelsta['lat'], lon1=trace.labelsta['lon'], lat2=event.srcloc[0], lon2=event.srcloc[1])[2] if rotate is True else 180
                        # azimuth = gps2dist_azimuth(lat1=-7.913, lon1=110.523, lat2=-0.660, lon2=133.430)[2]
                        
                        # rotate to TRZ coordinate
                        stream.rotate('NE->RT', back_azimuth=azimuth)

                    # save data
                    if not os.path.exists(savedir):
                        try: os.makedirs(savedir)
                        except FileExistsError: pass # sometimes happens when folder created by another subprocess
                        except: raise Exception(f"error when opening a new folder: {savedir}")
                    stream.write(f"{savedir}/{network_code}.{trace.labelsta['name']}.LH.obspy", format="PICKLE")
                    
                    # load p info
                    p_status = None
                    p_weight = None
                    p_travel_sec = None
                    is_calctim_defined = False

                    for record in trace.records:
                        if record.phase == 'P':
                            p_status = 'manual'
                            p_weight = 1/record.error
                            p_calctim0 = event.srctime + record.calctim
                            p_obstim0 = event.srctime + record.obstim
                            is_calctim_defined = True
                    
                    if not is_calctim_defined:
                        try: p_calctim0 = event.srctime + self.picker.model.get_travel_times(event.srcloc[2], trace.labelsta['dist'], ['P'])[0].time
                        except: p_calctim0 = event.srctime + self.picker.default_p_calctime

                    # resampling
                    shift_range = random.randrange(shift[0],shift[1]+1)
                    if resample != 0:
                        shift_bit = random.randrange(resample)
                        delta = 1/resample
                        stream.trim(starttime=fn_starttime_train(p_calctim0+shift_range), endtime=fn_endtime_train(p_calctim0+shift_range+1))
                        stream.interpolate(sampling_rate=resample, method='lanczos', a=20)
                        if len(stream) == 3:
                            stdshape = (int(1500*resample), 3)
                            waveform_data = np.transpose([np.array(stream[0].data[shift_bit:], dtype=np.float32), np.array(stream[1].data[shift_bit:], dtype=np.float32), np.array(stream[2].data[shift_bit:], dtype=np.float32)]) 
                            if waveform_data.shape[0] <= int(1520*resample) and waveform_data.shape[0] > int(1500*resample): waveform_data = waveform_data[:int(1500*resample),0:3]
                    else:
                        shift_bit = 0
                        delta = 1
                        stream.trim(starttime=p_calctim0-100+shift_range, endtime=p_calctim0+1400+shift_range)
                        stdshape = (1500, 3)
                        if len(stream) == 3:
                            waveform_data = np.transpose([np.array(stream[0].data, dtype=np.float64), np.array(stream[1].data, dtype=np.float64), np.array(stream[2].data, dtype=np.float64)]) 
                            if waveform_data.shape[0] <= 1520 and waveform_data.shape[0] > 1500: waveform_data = waveform_data[:1500,0:3]

                    # check for problematic values in array
                    waveform_data = waveform_data.astype(np.float32)
                    anynan = np.isnan(waveform_data).any()
                    # conditions = ((p_status or s_status) and waveform_data.shape==stdshape and not anynan)
                    # conditions = (p_status and s_status and waveform_data.shape==stdshape and not anynan) # conditions for training data
                    conditions = (waveform_data.shape==stdshape and not anynan) # conditions for raw data

                    if conditions:
                        # load s info
                        s_status = None
                        s_weight = None
                        s_travel_sec = None

                        for record in trace.records:
                            if record.phase == 'S':
                                s_status = 'manual'
                                s_weight = 1/record.error
                                s_obstim0 = event.srctime + record.obstim

                        # set up travel times and waveform info
                        if p_status:
                            p_travel_sec = p_obstim0 - fn_starttime_train(p_calctim0 + shift_range) - shift_bit * delta

                        if s_status:
                            s_travel_sec = s_obstim0 - fn_starttime_train(p_calctim0 + shift_range) - shift_bit * delta
                            coda_end_sample = int((s_travel_sec-60)/delta)
                            snr = (np.sum(abs(waveform_data[int((s_travel_sec-10)/delta):int((s_travel_sec+50)/delta),:]), axis=0) / (60/delta)) / (np.sum(abs(waveform_data[0:int(40/delta),:]), axis=0) / (40/delta))
                        elif p_status:
                            coda_end_sample = int((p_travel_sec+400)/delta)
                            snr = (np.sum(abs(waveform_data[int((p_travel_sec+20)/delta):int((p_travel_sec+400)/delta),:]), axis=0) / (380/delta)) / (np.sum(abs(waveform_data[0:int(40/delta),:]), axis=0) / (40/delta)) 
                        else:
                            coda_end_sample = None
                            snr = None
                        
                        # set up waveform attributes
                        waveform_attrs = {
                            'network_code': network_code,
                            'receiver_code': trace.labelsta['name'],
                            'receiver_type': 'LH',
                            'receiver_latitude': trace.labelsta['lat'], 
                            'receiver_longitude': trace.labelsta['lon'], 
                            'receiver_elevation_m': None, 
                            'p_arrival_sample': int(p_travel_sec/delta) if p_status else None, 
                            'p_status': p_status, 
                            'p_weight': p_weight, 
                            'p_travel_sec': p_travel_sec, 
                            's_arrival_sample': int(s_travel_sec/delta) if s_status else None, 
                            's_status': s_status, 
                            's_weight': s_weight, 
                            'source_id': 'None', 
                            'source_origin_time': str(event.srctime), 
                            'source_origin_uncertainty_sec': None, 
                            'source_latitude':event.srcloc[0], 
                            'source_longitude': event.srcloc[1], 
                            'source_error_sec': None, 
                            'source_gap_deg': None, 
                            'source_horizontal_uncertainty_km': None, 
                            'source_depth_km': event.srcloc[2], 
                            'source_depth_uncertainty_km': None, 
                            'source_magnitude': 0, 
                            'source_magnitude_type': None, 
                            'source_magnitude_author': None, 
                            'source_mechanism_strike_dip_rake': None, 
                            'source_distance_deg': trace.labelsta['dist'], 
                            'source_distance_km': trace.labelsta['dist'] * 111.1, 
                            'back_azimuth_deg': trace.labelsta['azi'], 
                            'snr_db': snr, 
                            'coda_end_sample': coda_end_sample, 
                            'trace_start_time': str(event.srctime), 
                            'trace_category': 'earthquake_local', 
                            'trace_name': event_name
                            }
                        
                        return {'data': waveform_data, 'attrs': waveform_attrs}
                    
                        # if conditions:
                        #     dataset = f.create_dataset(f"data/{event_name}",data=waveform_data)
                        #     if p_status:
                        #         p_travel_sec = p_obstim0 - fn_starttime_train(p_calctim0 + shift_range) - shift_bit * delta
                        #         dataset.attrs['p_arrival_sample'] = int(p_travel_sec/delta)
                        #         dataset.attrs['p_status'] = p_status
                        #         dataset.attrs['p_weight'] = p_weight
                        #         dataset.attrs['p_travel_sec'] = p_travel_sec
                        #     if s_status:
                        #         s_travel_sec = s_obstim0 - fn_starttime_train(p_calctim0 + shift_range) - shift_bit * delta
                        #         dataset.attrs['s_arrival_sample'] = int(s_travel_sec/delta)
                        #         dataset.attrs['s_status'] = s_status
                        #         dataset.attrs['s_weight'] = s_weight
                        #         coda_end_sample = int((s_travel_sec-60)/delta)
                        #         snr = (np.sum(abs(waveform_data[int((s_travel_sec-10)/delta):int((s_travel_sec+50)/delta),:]), axis=0) / (60/delta)) / (np.sum(abs(waveform_data[0:int(40/delta),:]), axis=0) / (40/delta))
                        #     else:
                        #         coda_end_sample = int((p_travel_sec+400)/delta)
                        #         snr = (np.sum(abs(waveform_data[int((p_travel_sec+20)/delta):int((p_travel_sec+400)/delta),:]), axis=0) / (380/delta)) / (np.sum(abs(waveform_data[0:int(40/delta),:]), axis=0) / (40/delta)) 
                            
                        #     dataset.attrs['coda_end_sample'] = coda_end_sample
                        #     dataset.attrs['snr_db'] = snr
                        #     dataset.attrs['trace_category'] = 'earthquake_local'
                        #     dataset.attrs['network_code'] = network_code
                        #     dataset.attrs['source_id'] = 'None'
                        #     dataset.attrs['source_distance_km'] = trace.labelsta['dist'] * 111.1
                        #     dataset.attrs['trace_name'] = event_name      
                        #     dataset.attrs['trace_start_time'] = str(event.srctime)
                        #     dataset.attrs['source_magnitude'] = 0
                        #     dataset.attrs['receiver_type'] = 'LH'
                        #     return {'network_code': network_code, 'receiver_code': trace.labelsta['name'], 'receiver_type': 'LH', 'receiver_latitude': trace.labelsta['lat'], 'receiver_longitude': trace.labelsta['lon'], 'receiver_elevation_m': None, 'p_arrival_sample': int(p_travel_sec/delta) if p_status else None, 'p_status': p_status, 'p_weight': p_weight, 'p_travel_sec': p_travel_sec, 's_arrival_sample': int(s_travel_sec/delta) if s_status else None, 's_status': s_status, 's_weight': s_weight, 'source_id': None, 'source_origin_time': event.srctime, 'source_origin_uncertainty_sec': None, 'source_latitude':event.srcloc[0], 'source_longitude': event.srcloc[1], 'source_error_sec': None, 'source_gap_deg': None, 'source_horizontal_uncertainty_km': None, 'source_depth_km': event.srcloc[2], 'source_depth_uncertainty_km': None, 'source_magnitude': None, 'source_magnitude_type': None, 'source_magnitude_author': None, 'source_mechanism_strike_dip_rake': None, 'source_distance_deg': trace.labelsta['dist'], 'source_distance_km': trace.labelsta['dist'] * 111.1, 'back_azimuth_deg': trace.labelsta['azi'], 'snr_db': snr, 'coda_end_sample': [[coda_end_sample]], 'trace_start_time': event.srctime, 'trace_category': 'earthquake_local', 'trace_name': event_name}
                except UnboundLocalError:
                    print(f"Unbound local error for {obsfile_name}: {e}")
                except ValueError as e:
                    print(f"Value error for {obsfile_name}: {e}")
                except FileNotFoundError as e:
                    print(f"File-not-found error for {obsfile_name}: {e}")
                # except:
                #     print(f"Unexpect error for {obsfile_name}")
                    
    def _prepare_resplist_par(self, item, respdir):
        print(f"preparing instrument {str(item[0])}")
        return (item[0], Instrument(station_str=item[0], event_records=item[1], respdir=respdir))
    
    def prepare_resplist(self, respdir='/Users/jun/phasepick/resp_catalog'):
        if not os.path.exists(self.resp_list_filename):
            with open(self.station_list_filename, 'rb') as infile:
                stalist = pickle.load(infile)
                with ThreadPool(cpu_count()) as p:
                    self.resplist = dict(p.starmap(self._prepare_resplist_par, zip(stalist.items(), repeat(respdir))))
                with open(self.resp_list_filename, 'wb') as outfile:
                    stalist = pickle.dump(self.resplist, outfile)
        else:
            with open(self.resp_list_filename, 'rb') as file:
                self.resplist = pickle.load(file)

    def get_datalist(self, resample=0, rotate=True, preprocess=True, shift=(-100,100), output='./test.hdf5', overwrite_hdf=True, obsfile='separate', year_option=None, dir_ext='', cpu_number=36, respdir='/Users/jun/phasepick/resp_catalog'):
        if not shift: shift = (0,0)
        
        # get instrument response for reference station
        reference_responses = [InstrumentResponse(network='SR', station='GRFO', component=component, timestamp=UTCDateTime(1983, 1, 1)) for component in ['LHE', 'LHN', 'LHZ']]

        # load station list and build inventory for station response
        if preprocess: self.prepare_resplist(respdir)

        # set directory
        loaddir = f'rawdata{dir_ext}' if preprocess else 'training'

        # open new hdf5 database
        with h5py.File(output, 'w' if overwrite_hdf else 'a') as f:
            if overwrite_hdf: f.create_group("data")

            # create datalist by searching all record lists
            datalist = []
            print(f"preparing waveforms in parallel, #cpu={cpu_number}.")

            for event in self.events:
                if year_option:
                    if event.srctime.year != year_option: continue
                
                if event.srctime.precision != 3: event.srctime.precision = 3
                if preprocess=='bandpass': savedir = f"./training_bandpass{dir_ext}/{event.srctime}"
                elif preprocess=='onlyrot': savedir = f"./training_onlyrot{dir_ext}/{event.srctime}"
                elif preprocess: savedir = f"./training{dir_ext}/{event.srctime}"

                if (not overwrite_hdf) and os.path.exists(savedir): continue
                    
                if obsfile == 'compiled': obsfilenames = [f"./{loaddir}/{event.srctime}.LH.obspy"]
                else: obsfilenames = None

                print(f"processing for {len(event.stations)} stations at {event.srctime}...")
                
                # set up parameters for parallel computing
                sublist = []
                common_args = {
                    'event': event,
                    'obsfile': obsfile,
                    'preprocess': preprocess,
                    'resample': resample,
                    'shift': shift,
                    'rotate': rotate,
                    'reference_responses': reference_responses,
                    'loaddir': loaddir,
                    'obsfilenames': obsfilenames,
                    'savedir': savedir,
                    'output': output
                }

                # start parallel computing for preprocessing
                with ThreadPool(cpu_number) as p:
                    sublist = p.starmap(self._par_datalist,
                        zip(event.stations, repeat(common_args)))
            
                # retreive data and attributes from each event
                for item in sublist:
                    if item:
                        dataset = f.create_dataset(f"data/{item['attrs']['trace_name']}", data=item['data'])
                        for attr, val in item['attrs'].items():
                            if not val is None: dataset.attrs[attr] = val
                        item['attrs']['source_origin_time'] = UTCDateTime(item['attrs']['source_origin_time'])
                        item['attrs']['trace_start_time'] = UTCDateTime(item['attrs']['trace_start_time'])
                        item['attrs']['coda_end_sample'] = [[item['attrs']['coda_end_sample']]]
                        datalist.append(item['attrs'])
                print(f"All waveforms at {event.srctime} are done.")

        return datalist
    
    def load_datalist_from_hdf5(self, hdf5_filename):
        with h5py.File(hdf5_filename, 'r') as f:
            datalist = []
            for item in list(f['data'].keys()):
                attrs_dict = dict(item.attrs)
                attrs_dict['source_origin_time'] = UTCDateTime(attrs_dict['source_origin_time'])
                attrs_dict['trace_start_time'] = UTCDateTime(attrs_dict['trace_start_time'])
                attrs_dict['coda_end_sample'] = [[attrs_dict['coda_end_sample']]]
                datalist.append(attrs_dict)
        return datalist


    def __init__(self, picker, client: Client, paths: list, autofetch=False, isTable=True, station_list_filename="/Users/jun/phasepick/stalist.pkl", resp_list_filename="/Users/jun/phasepick/resp_catalog/resplist.pkl"):
        self.picker = picker
        self.client = client
        self.numdownloaded = 0
        self.station_list_filename = station_list_filename
        self.resp_list_filename = resp_list_filename

        # create event list from paths
        self.events = []
        for path in paths:
            if isTable:
                numevent = self._table2events(path)
                print(f"table read successfully, {numevent} events added")
            else:
                numevent = self._folder2events(path)
                print(f"folder read successfully, {numevent} events added")

        # fetch data if autofetch toggled
        if autofetch: self.fetch(skip_existing_events=False)
    

class Picker():
    def __init__(self, default_p_calctime=450):
        self.client = Client('IRIS')
        self.station_list = None
        self.station_dict = None
        self.default_p_calctime = default_p_calctime   

    def create_dataset(self, table_filenames):
        "create dataset from scretch"
        self.data = SeismicData(self, self.client, table_filenames, autofetch=False)

    def create_dataset_from_folder(self, folder_paths):
        "create dataset from scretch"
        self.data = SeismicData(self, self.client, folder_paths, autofetch=False, isTable=False)

    def load_dataset(self, filename, verbose=False):
        "load existing dataset"
        with open(filename, 'rb') as file:
            loaddata = pickle.load(file)
        self.data = SeismicData(self, self.client, [])
        self.data.events = loaddata.events
        # debug message
        if verbose: 
            printphase = [record.phase for record in self.data.events[0].stations[3].records]
            print(f"loaded dataset has {len(self.data.events)} events")
            print(f"#1 event has {len(self.data.events[0])} stations, records of #4 event: {', '.join(printphase)}")
    
    def dump_dataset(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.data, file, pickle.HIGHEST_PROTOCOL)

    def get_stationlist(self):
        station_list = []
        station_dict = {}
        for event in self.data.events:
            for station in event.stations:
                if not station.labelsta['name'] in station_list:
                    station_list.append(station.labelsta['name'])
                    station_dict[station.labelsta['name']] = station
        self.station_list = station_list
        self.station_dict = station_dict
        return self.station_list, self.station_dict
    

    def _resampling(st):
        need_resampling = [tr for tr in st if tr.stats.sampling_rate != resample_rate]
        if len(need_resampling) > 0:
        # print('resampling ...', flush=True)    
            for indx, tr in enumerate(need_resampling):
                if tr.stats.delta < 1/resample_rate:
                    tr.filter('lowpass',freq=resample_rate*0.45,zerophase=True)
                tr.resample(resample_rate)
                tr.stats.sampling_rate = resample_rate
                tr.stats.delta = 1/resample_rate
                tr.data.dtype = 'float32' #'int32'
                st.remove(tr)
                st.append(tr)
                
        return st
    
    # def _prepare_picking_par(self, station_code, overlap=0, obsfile='separate'):
    def _prepare_picking_par(self, station_code, overlap=0, obsfile='separate'):
        output_name = station_code
        station = self.station_dict[station_code]
        
        try:
            os.remove(output_name+'.hdf5')
            os.remove(output_name+".csv")
        except Exception:
            pass
        
        HDF = h5py.File(f'{self.save_dir}/{output_name}.hdf5', 'a')
        HDF.create_group("data")
    
        csvfile = open(f'{self.save_dir}/{output_name}.csv', 'w')
        output_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['trace_name', 'start_time'])
        csvfile.flush()   
    
        if obsfile=='separate':
            filenames = glob.glob(f'{self.waveform_dir}/*/*.{station_code}.*') #[join(station, ev) for ev in listdir(station) if ev.split("/")[-1] != ".DS_Store"];
        else:
            # filenames = glob.glob(f'{self.waveform_dir}/*.{station_code}.*')
            filenames = glob.glob(f'{self.waveform_dir}/2010*.{station_code}.*')
        
        time_slots, comp_types = [], []
        
        print('============ Station {} has {} chunks of data.'.format(station_code, len(filenames)), flush=True)  
            
        count_chuncks=0; c1=0; c2=0; c3=0
        
        for filename in filenames:
            st = read(filename, debug_headers=True)
            component_num = len(st)

            if obsfile=='separate':
                if component_num == 3:
                    count_chuncks += 1; c3 += 1
                    org_samplingRate = st[0].stats.sampling_rate
            
                    time_slots.append((st[0].stats.starttime, st[0].stats.endtime))
                    comp_types.append(3)
                    # print('  * '+station_code+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 1 components .. sampling rate: '+str(org_samplingRate)) 
                    
                    if len([tr for tr in st if tr.stats.sampling_rate != resample_rate]) != 0:
                        try:
                            st.interpolate(sampling_rate=resample_rate, method='lanczos', a=20)
                        except Exception:
                            st=self._resampling(st) 
                    
                    # longest = st[0].stats.npts
                    # start_time = st[0].stats.starttime
                    # end_time = st[0].stats.endtime
                    
                    # for tt in st:
                    #     if tt.stats.npts > longest:
                    #         longest = tt.stats.npts
                    #         start_time = tt.stats.starttime
                    #         end_time = tt.stats.endtime

                    ####
                    target_srctime = UTCDateTime(filename.split('/')[-2])
                    p_calctim = self.default_p_calctime
                    for event in self.data.events:
                        if event.srctime == target_srctime:
                            ref_trace = event._findstation(station)
                            try:
                                p_calctim = self.model.get_travel_times(event.srcloc[2], ref_trace.labelsta['dist'], ['P'])[0].time
                                print("Using calculated P arrival window for", output_name, filename)
                            except:
                                print("Using default P arrival window for", output_name, filename)
                            break
                        
                    start_time = fn_starttime_train(target_srctime+p_calctim)
                    end_time = fn_endtime_train(target_srctime+p_calctim)
                    st.trim(start_time, end_time, pad=True, fill_value=0)
                    # print(filename, target_srctime, p_calctim, start_time)
                    ####

                    chanL = [st[0].stats.channel[-1], st[1].stats.channel[-1], st[2].stats.channel[-1]]
                    w = st.slice(start_time, start_time+1500)                    
                    npz_data = np.zeros([6000,3])
                    
                    try:
                        npz_data[:,2] = w[chanL.index('Z')].data[:6000]
                        npz_data[:,0] = w[chanL.index('T')].data[:6000]
                        npz_data[:,1] = w[chanL.index('R')].data[:6000]
                    
                        tr_name = st[0].stats.station+'_'+st[0].stats.network+'_'+st[0].stats.channel[:2]+'_'+filename.split('/')[-2]
                        HDF = h5py.File(f'{self.save_dir}/{output_name}.hdf5', 'r')
                        dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                        dsF.attrs["trace_name"] = tr_name
                        dsF.attrs["receiver_code"] = station_code
                        dsF.attrs["network_code"] = st[0].stats.network
                        dsF.attrs["receiver_latitude"] = station.labelsta['lat'] #if station.labelsta['lat'] else st[0].stats.network
                        dsF.attrs["receiver_longitude"] = station.labelsta['lon'] #if station.labelsta['lon'] else st[0].stats.network
                        # dsF.attrs["receiver_elevation_m"] = 0
                        dsF.attrs["sampling_rate"] = resample_rate
                            
                        start_time_str = str(start_time)   
                        start_time_str = start_time_str.replace('T', ' ')                 
                        start_time_str = start_time_str.replace('Z', '')          
                        dsF.attrs['trace_start_time'] = start_time_str
                        HDF.flush()
                        output_writer.writerow([str(tr_name), start_time_str])  
                        csvfile.flush()   
                    except:
                        print("Failed to write data for", output_name, filename)
                else: print(output_name, filename, component_num)                    
            else:
                # get list for all stations with records
                station_list = []
                for tr in st:
                    if not tr.stats.station in station_list:
                        if len(st.select(station=tr.stats.station)) == 3:
                            station_list.append(tr.stats.station)
                
                # loop for each station in the list
                for station_name in station_list:
                    st_select = st.select(station=station_name)

                    count_chuncks += 1; c3 += 1
                    org_samplingRate = st_select[0].stats.sampling_rate
            
                    time_slots.append((st_select[0].stats.starttime, st_select[0].stats.endtime))
                    comp_types.append(3)
                    # print('  * '+station_code+' ('+str(count_chuncks)+') .. '+month.split('T')[0]+' --> '+month.split('__')[1].split('T')[0]+' .. 1 components .. sampling rate: '+str(org_samplingRate)) 
                    
                    if len([tr for tr in st_select if tr.stats.sampling_rate != resample_rate]) != 0:
                        try:
                            st_select.interpolate(sampling_rate=resample_rate, method='lanczos', a=20)
                        except Exception:
                            st_select=self._resampling(st_select) 
                    
                    target_srctime = UTCDateTime(filename.split('/')[-2])
                    p_calctim = self.default_p_calctime
                    for event in self.data.events:
                        if event.srctime == target_srctime:
                            ref_trace = event._findstation(station)
                            try:
                                p_calctim = self.model.get_travel_times(event.srcloc[2], ref_trace.labelsta['dist'], ['P'])[0].time
                                print("Using calculated P arrival window for", output_name, filename)
                            except:
                                print("Using default P arrival window for", output_name, filename)
                            break
                        
                    start_time = fn_starttime_train(target_srctime+p_calctim)
                    end_time = fn_endtime_train(target_srctime+p_calctim)
                    st_select.trim(start_time, end_time, pad=True, fill_value=0)
                    # print(filename, target_srctime, p_calctim, start_time)
                    ####

                    chanL = [st_select[0].stats.channel[-1], st_select[1].stats.channel[-1], st_select[2].stats.channel[-1]]
                    w = st_select.slice(start_time, start_time+1500)                    
                    npz_data = np.zeros([6000,3])
                    
                    try:
                        npz_data[:,2] = w[chanL.index('Z')].data[:6000]
                        npz_data[:,0] = w[chanL.index('T')].data[:6000]
                        npz_data[:,1] = w[chanL.index('R')].data[:6000]
                    
                        tr_name = st_select[0].stats.station+'_'+st_select[0].stats.network+'_'+st_select[0].stats.channel[:2]+'_'+filename.split('/')[-2]
                        HDF = h5py.File(f'{self.save_dir}/{output_name}.hdf5', 'r')
                        dsF = HDF.create_dataset('data/'+tr_name, npz_data.shape, data = npz_data, dtype= np.float32)        
                        dsF.attrs["trace_name"] = tr_name
                        dsF.attrs["receiver_code"] = station_code
                        dsF.attrs["network_code"] = st_select[0].stats.network
                        dsF.attrs["receiver_latitude"] = station.labelsta['lat'] #if station.labelsta['lat'] else st[0].stats.network
                        dsF.attrs["receiver_longitude"] = station.labelsta['lon'] #if station.labelsta['lon'] else st[0].stats.network
                        # dsF.attrs["receiver_elevation_m"] = 0
                        dsF.attrs["sampling_rate"] = resample_rate
                            
                        start_time_str = str(start_time)   
                        start_time_str = start_time_str.replace('T', ' ')                 
                        start_time_str = start_time_str.replace('Z', '')          
                        dsF.attrs['trace_start_time'] = start_time_str
                        HDF.flush()
                        output_writer.writerow([str(tr_name), start_time_str])  
                        csvfile.flush()   
                    except:
                        print("Failed to write data for", output_name, filename)
                else: print(output_name, filename, component_num)               
                    
            st = None
                
        HDF.close() 
        
        dd = pd.read_csv(f'{self.save_dir}/{output_name}.csv')
                
        
        # assert count_chuncks == len(filenames)  
        # assert sum(slide_estimates)-(fln/100) <= len(dd) <= sum(slide_estimates)+10
        self.data_track[output_name]=[time_slots, comp_types]
        print(f" Station {output_name} had {len(filenames)} chuncks of data") 
        print(f"{len(dd)} slices were written, {len(filenames)} were expected.")
        print(f"Number of 1-components: {c1}. Number of 2-components: {c2}. Number of 3-components: {c3}.")
        try:
            print(f"Original samplieng rate: {org_samplingRate}.") 
            self.repfile.write(f' Station {output_name} had {len(filenames)} chuncks of data, {len(dd)} slices were written, {int(len(filenames))} were expected. Number of 1-components: {c1}, Number of 2-components: {c2}, number of 3-components: {c3}, original samplieng rate: {org_samplingRate}\n')
        except Exception:
            pass

    def prepare_catalog(self, waveform_dir: str, preproc_dir: str, save_dir: str, n_processor=None):
        'Prepare hdf for predictor function following EQTransformer (original script by mostafamousavi) for catalog events.'
        self.waveform_dir = waveform_dir
        self.preproc_dir = preproc_dir
        self.save_dir = save_dir
        if os.path.isdir(save_dir):
            print(f' *** " {save_dir} " directory already exists!')
            inp = input(" * --> Do you want to create a new empty folder? Type (Yes or y) ")
            if inp.lower() == "yes" or inp.lower() == "y":        
                shutil.rmtree(save_dir)  
        os.makedirs(save_dir)

        if not os.path.exists(preproc_dir):
            os.makedirs(preproc_dir)
        self.repfile = open(os.path.join(preproc_dir,"prepare_report.txt"), 'w')

        if self.station_dict is None or self.station_list is None:
            self.get_stationlist()

        self.model = TauPyModel(model="prem")
        self.data_track = dict()

        if not n_processor:
            n_processor = cpu_count()
        with ThreadPool(n_processor) as p:
            p.map(self._prepare_picking_par, self.station_list) 
        with open(os.path.join(preproc_dir,'time_tracks.pkl'), 'wb') as f:
            pickle.dump(self.data_track, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    table_dir = "/Users/jun/Downloads/drive-download-20220512T014633Z-001"
    table_filenames = [f"{table_dir}/Pcomb.4.07.09.table",f"{table_dir}/Scomb.4.07.09.table"]
    resample_rate = 4.0
    
    # # create dataset from scretch and dump
    # picker = Picker()
    # picker.create_dataset(table_filenames)
    # picker.dump_dataset("data.pkl")
    
    # # create dataset from scretch, fetch seismic data, and dump
    # picker = Picker()
    # picker.create_dataset(table_filenames)
    # picker.data.fetch()
    # picker.dump_dataset("data_fetched.pkl")

    # # create dataset and link existing seismic data in local device
    # picker = Picker()
    # picker.create_dataset(table_filenames)
    # picker.data.link_downloaded(israwdata=True)
    # picker.dump_dataset("data_fetched.pkl")

    # # load fetched dataset, remove instrument response, bandpass frequency, and create training dataset
    # picker = Picker()
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess='bandpass', output='./upbpANMO_shift.hdf5')
    # random.shuffle(datalist)
    # df = pd.DataFrame(datalist)
    # df.to_csv('training_PandS_upbpANMO_shift.csv', index=False)
    
    # load fetched dataset, remove instrument response, bandpass frequency, and create training dataset
    # picker = Picker()
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess='onlyrot', output='./uprotANMO_shift.hdf5')
    # random.shuffle(datalist)
    # df = pd.DataFrame(datalist)
    # df.to_csv('training_PandS_uprotANMO_shift.csv', index=False)
    
    # # load fetched dataset, remove instrument response, and create training dataset
    # picker = Picker()
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess=False, output='./updeANMO_shift.hdf5')
    # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess=False, shift=False, output='./updeANMO.hdf5')
    # random.shuffle(datalist)
    # df = pd.DataFrame(datalist)
    # df.to_csv('training_PandS_updeANMO.csv', index=False)

    # # train a model with EQTransformer
    # model_name="updeANMO_shift"
    # trainer_name=f"test_trainer_{model_name}"
    # tester_name=f"test_tester_{model_name}"
    # eqt.trainer(input_hdf5=f'{model_name}.hdf5', input_csv=f'training_PandS_{model_name}.csv', output_name=trainer_name,
    #     cnn_blocks=2, lstm_blocks=1, padding='same', activation='relu', drop_rate=0.2, label_type='gaussian',
    #     add_event_r=0.6, add_gap_r=0.2, shift_event_r=0.9, add_noise_r=0.5, mode='generator',
    #     train_valid_test_split=[0.60, 0.20, 0.20], batch_size=20, epochs=10, patience=2, pre_emphasis=True,#pre_emphasis=False
    #     gpuid=None, gpu_limit=None, input_dimention=(6000, 3))
    # eqt.tester(input_hdf5=f'{model_name}.hdf5', input_testset=f'{trainer_name}_outputs/test.npy', 
    #            input_model=f'{trainer_name}_outputs/final_model.h5', output_name=tester_name,
    #            detection_threshold=0.20, P_threshold=0.1, S_threshold=0.1, number_of_plots=30, estimate_uncertainty=True, number_of_sampling=5,
    #            input_dimention=(6000, 3), normalization_mode='std', mode='generator', batch_size=10, gpuid=None, gpu_limit=None)

    # # load dataset and prepare prediction data
    # picker = Picker(default_p_calctime=450)
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # picker.prepare_catalog('./training', './hmsl_preproc', './hmsl_hdfs', 10)

    # # load dataset and prepare prediction data for banpass
    # picker = Picker(default_p_calctime=450)
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # picker.prepare_catalog('./training_bandpass', './hmsl_bp_preproc', './hmsl_bp_hdfs', 10)

    # load dataset and prepare prediction data for banpass
    # picker = Picker(default_p_calctime=450)
    # picker.load_dataset('data_fetched.pkl', verbose=True)
    # picker.prepare_catalog('./training_onlyrot', './hmsl_rot_preproc', './hmsl_rot_hdfs', 10)

    # # create dataset from scretch, fetch seismic data, and dump
    # picker = Picker()
    # picker.create_dataset([])
    # catalog = np.load('/Users/jun/phasepick/gcmt.npy',allow_pickle=True)
    # picker.data.events = catalog
    # picker.data.fetch()
    # picker.dump_dataset("./rawdata_catalog2/data_fetched_catalog.pkl")

    # # load fetched dataset, remove instrument response, and create training dataset
    # picker = Picker()
    # picker.load_dataset('./rawdata_catalog2/data_fetched_catalog_2010.pkl', verbose=True)
    # datalist = picker.data.get_datalist(resample=resample_rate, preprocess=True, output='./catalog_2010_preproc.hdf5', overwrite_hdf=False, obsfile="compiled", year_option=2010, dir_ext='_catalog2')
    # df = pd.DataFrame(datalist)
    # df.to_csv('catalog_2010_preproc.csv', index=False)
    # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess=False, shift=False, output='./updeANMO.hdf5', obsfile="compiled", year_option=2010, dir_ext='_catalog2')
    # random.shuffle(datalist)
    # df = pd.DataFrame(datalist)
    # df.to_csv('training_PandS_updeANMO.csv', index=False)

    # load dataset and prepare prediction data
    picker = Picker(default_p_calctime=450)
    picker.load_dataset('./rawdata_catalog2/data_fetched_catalog_2010.pkl', verbose=True)
    # picker.prepare_catalog('./training_catalog2', '/Volumes/seismic/catalog_preproc', '/Volumes/seismic/catalog_hdfs')
    picker.data.prepare_resplist()

    # picker = Picker()
    # picker.load_dataset('data_fetched_catalog.pkl', verbose=False)
    # summ=0
    # for ev in picker.data.events:
    #     summ+=len(ev.stations)
    # print(summ)
