import io
import os
import sys
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
from multiprocessing import Manager, cpu_count, get_context
from multiprocessing.pool import ThreadPool, Pool
from scipy.fft import rfft, irfft
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy import read, read_inventory, Stream
from obspy.taup import TauPyModel
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
from obspy.signal.invsim import simulate_seismometer, evalresp_for_frequencies
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import InternalMSEEDError
from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
from obspy.core.util.obspy_types import ObsPyException
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# if len(sys.argv)==2: log = open(sys.argv[1], "w"); sys.stdout = log; sys.stderr = log
# warnings.simplefilter('ignore', category=ObsPyDeprecationWarning)
warnings.simplefilter('ignore')

fn_starttime_full = lambda srctime: srctime - 0.5 * 60 * 60
fn_endtime_full = lambda srctime: srctime + 2 * 60 * 60
fn_starttime_train = lambda srctime: srctime - 250
fn_endtime_train = lambda srctime: srctime + 1250

def split_array(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def flatten_list(l):
    for el in l:
        if isinstance(el, list): yield from flatten_list(el)
        else: yield el

class Event():
    def __init__(self, srctime: UTCDateTime, lat, lon, dep, mag=None, gcmtid=None, hypotime=None):
        self.srctime = srctime
        self.srcloc = (lat, lon, dep)
        self.magnitude = mag
        self.gcmtid = gcmtid
        self.stations = []
        self.hypotime = hypotime if hypotime else srctime
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
    def __repr__(self):
        if 'magnitude' in dir(self):
            return f"{self.srctime} magnitude {self.magnitude} at lat. {self.srcloc[0]} lon. {self.srcloc[1]}"
        else:
            return f"{self.srctime} magnitude unknown at lat. {self.srcloc[0]} lon. {self.srcloc[1]}"

class Station():
    def __init__(self, station, lat, lon, dist, azi, loc=''):
        self.labelsta = {'name': station, 'lat': lat, 'lon': lon, 'dist': dist, 'azi': azi, 'loc': loc}
        self.labelnet = {'code': None}
        self.records = []
        self.isdataexist = False
    def __eq__(self, target):
        return (self.labelsta['name'] == target.labelsta['name'])
    def __ne__(self, target):
        return not self.__eq__(target)
    def __repr__(self):
        return f"observation at station {self.labelsta['name']} of {self.labelnet['code'] or 'UNKNOWN'} network with azimuth angle {(360-self.labelsta['azi'])} deg"

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
    def download_inventory(savepath, station_str, **kwargs) -> dict:
        client = Client("IRIS")
        inv = client.get_stations(level='RESP', **kwargs)
        startdate = inv[0][0].start_date
        enddate = inv[0][0].end_date or UTCDateTime("30010101")
        output_filename = f"{savepath}/{station_str}.{startdate.year}.{startdate.julday:03}.{startdate.hour:02}.{enddate.year}.{enddate.julday:03}.{enddate.hour:02}"
        inv.write(output_filename, format="STATIONXML")
        return {'filename': output_filename, 'start_date': startdate, 'end_date': enddate}
    
    def find_obspy_station(self, datetime=None, return_instance=True):
        """This function returns a dictionary of station objects with all available responses that cover the given timestamp."""
        if type(datetime) is str: datetime = UTCDateTime(datetime)

        # get timestamp from first response of each set in the response inventory
        for station_dict in self.resp_inventory:
            # return responses if found in resp_inventory
            if datetime > station_dict['start_date'] and (not station_dict['end_date'] or datetime < station_dict['end_date']):
                return read_inventory(station_dict['filename'])[0][0] if return_instance else station_dict

        # download responses if not found in resp_inventory
        try:
            station_dict = self.download_inventory(self.respdir, f"{self.network_code}.{self.station_code}", network=self.network_code, station=self.station_code, starttime=datetime)
            self.resp_inventory.append(station_dict)
            return read_inventory(station_dict['filename'])[0][0] if return_instance else station_dict
        except:
            print(f"No data: {self.network_code}.{self.station_code} {datetime}")
            return None

    def add_event(self, datetime_str):
        self.resp4record[str(datetime_str)] = self.find_obspy_station(datetime_str, return_instance=False)

    def prepare_inventory(self, filenames):
        inventory = []
        for filename in filenames:
            inv = read_inventory(filename)
            startdate = inv[0][0].start_date
            enddate = inv[0][0].end_date or UTCDateTime("30010101")
            inventory.append({'filename': filename, 'start_date': startdate, 'end_date': enddate})
        return inventory

    def __init__(self, station_str=str, event_records=list, respdir=str):
        self.respdir = respdir
        if not os.path.exists(self.respdir): os.mkdir(self.respdir)
        #print(station_str)
        self.network_code, self.station_code, _ = station_str.split('.')
        self.resp4record = {}
        self.resp_inventory = self.prepare_inventory(glob.glob(f"{self.respdir}/{self.network_code}.{self.station_code}.*"))

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

        for datetime_str in event_records: self.add_event(datetime_str)

    def __repr__(self):
        return f"{self.network_code}.{self.station_code}: "+', '.join([f"{sta['start_date'].year}.{sta['start_date'].julday:03}.{sta['start_date'].hour:02}.{(sta['end_date'].year if sta['end_date'] else 3001)}.{(sta['end_date'].julday if sta['end_date'] else 1):03}.{(sta['end_date'].hour if sta['end_date'] else 0):02}" for sta in self.resp_inventory])
  
def _prepare_event_table_spawn(event, args):
    srctime = event.srctime
    srctime.precision = 3
    filename = f"{args['rawdata_dir']}/{srctime}.LH.obspy"

    if (os.path.exists(filename)):
        loaded_stream = read(filename)
        print(f"preparing event {srctime} at {filename}")
        processed_station = []
        for trace in loaded_stream:    
            try:
                # process only for a new station
                if not trace.stats.station in processed_station:
                    station = trace.meta.station
                    network = trace.meta.network
                    station_str = f'{network}.{station}.LH'
                    location_priority = ['', '00', None]
                    for location_matching in location_priority:
                        traces_matching = loaded_stream.select(station=station, location=location_matching)
                        if len(traces_matching) == 3: break
                        elif len(traces_matching) < 3: continue
                        else:
                            if len(traces_matching.select(channel='LHE'))>0 and len(traces_matching.select(channel='LHN'))>0:
                                traces_matching.remove(traces_matching.select(channel='LH1'))
                                traces_matching.remove(traces_matching.select(channel='LH2'))
                                if len(traces_matching) == 3: break
                            else:
                                traces_matching.remove(traces_matching.select(channel='LHN'))
                                traces_matching.remove(traces_matching.select(channel='LHE'))
                                if len(traces_matching) == 3: break
                    
                    if location_matching is None:
                        print(srctime, trace.stats.station, '... X (more or less than 3 LH components)')
                    else:
                        try:
                            stations_matching = args['resplist'][station_str].find_obspy_station(UTCDateTime(srctime))
                        # except: stations_matching = None
                        # try:
                            # if stations_matching == None: stations_matching = client.get_stations(level='station', network=network, station=station, starttime=starttime)[0][0]
                            if stations_matching == None: raise Exception("no matched station")
                            fetched = Station(stations_matching.code, stations_matching.latitude, stations_matching.longitude,
                                                dist=locations2degrees(lat1=stations_matching.latitude, long1=stations_matching.longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                                                azi=gps2dist_azimuth(lat1=stations_matching.latitude, lon1=stations_matching.longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2],
                                                loc=location_priority)
                            fetched.labelnet['code'] = network
                            fetched.isdataexist = True
                            event.stations.append(fetched)
                        except: print(srctime, trace.stats.station, '... X (failed setting up station)')
            except:
                    print(srctime, trace.stats.station, '... X (error in reading fetched traces)')
            processed_station.append(trace.stats.station)

        # time.sleep(0.2)
        print(srctime, f"Loaded fetched traces for {len(event.stations)} stations")
    else: print(f"cannot find event at {filename}")
    return event

def deconv_resp(rawdata, station_resps, reference_resps, working_freqencies=None):
    if working_freqencies is None: working_freqencies = np.linspace(1e-10, 0.5, len(rfft(rawdata[0].data)))
    proceed = rawdata.copy()

    # loop for all (three) components
    for i in range(len(station_resps)):
        # fast-fourier-transform the trace
        fdomain_data = rfft(rawdata[i].data)

        # get response array from frequencies
        sta_resp_interp = station_resps[i].get_evalresp_response_for_frequencies(working_freqencies)
        ref_resp_interp = reference_resps[i].get_evalresp_response_for_frequencies(working_freqencies)

        # deconvolve and convolve the trace
        fdomain_data = fdomain_data / sta_resp_interp * ref_resp_interp
        fdomain_data[0] = complex(0, 0)

        # inverse fast-fourier-transform the trace
        deconvolved_data = irfft(fdomain_data)
        proceed[i].data = deconvolved_data / max(deconvolved_data)

    return proceed

class SeismicData():
    def _findevent(self, target):
        for event in self.events:
            if event == target: return event
        return None
    def get_event_by_centroid_time(self, centroid_time: UTCDateTime):
        for event in self.events:
            if event.srctime == centroid_time: return event
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

    def _get_srctime(self, srctime):
        srctime.precision = 3
        starttime = srctime - 0.5 * 60 * 60
        endtime = srctime + 2 * 60 * 60
        return srctime, starttime, endtime

    def _prepare_event_table_par(self, event):
        srctime, starttime, endtime = self._get_srctime(event.srctime)
        filename = f"{self.rawdata_dir}/{srctime}.LH.obspy"
        # client = Client("IRIS")

        if (os.path.exists(filename)):
            loaded_stream = read(filename)
            print(f"preparing event {srctime} at {filename}")
            processed_station = []
            for trace in loaded_stream:    
                try:
                    # process only for a new station
                    if not trace.stats.station in processed_station:
                        station = trace.meta.station
                        network = trace.meta.network
                        station_str = f'{network}.{station}.LH'
                        location_priority = ['', '00', None]
                        for location_matching in location_priority:
                            traces_matching = loaded_stream.select(station=station, location=location_matching)
                            if len(traces_matching) == 3: break
                            elif len(traces_matching) > 3:
                                if len(traces_matching.select(channel='LHE'))>0 and len(traces_matching.select(channel='LHN'))>0:
                                    traces_matching.remove(traces_matching.select(channel='LH1'))
                                    traces_matching.remove(traces_matching.select(channel='LH2'))
                                    if len(traces_matching) == 3: break
                                else:
                                    traces_matching.remove(traces_matching.select(channel='LHN'))
                                    traces_matching.remove(traces_matching.select(channel='LHE'))
                                    if len(traces_matching) == 3: break
                        
                        if location_matching is None:
                            print(srctime, trace.stats.station, '... X (more or less than 3 LH components)')
                        else:
                            try: stations_matching = self.resplist[station_str].find_obspy_station(UTCDateTime(srctime))
                            except: stations_matching = None
                            try:
                                # if stations_matching == None: stations_matching = client.get_stations(level='station', network=network, station=station, starttime=starttime)[0][0]
                                if stations_matching == None: raise Exception("no matched station")
                                fetched = Station(stations_matching.code, stations_matching.latitude, stations_matching.longitude,
                                                    dist=locations2degrees(lat1=stations_matching.latitude, long1=stations_matching.longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                                                    azi=gps2dist_azimuth(lat1=stations_matching.latitude, lon1=stations_matching.longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2],
                                                    loc=location_matching)
                                fetched.labelnet['code'] = network
                                fetched.isdataexist = True
                                event.stations.append(fetched)
                            except: print(srctime, trace.stats.station, '... X (failed setting up station)')
                        processed_station.append(trace.stats.station)
                except:
                        print(srctime, trace.stats.station, '... X (error in reading fetched traces)')
                        processed_station.append(trace.stats.station)

            # time.sleep(0.2)
            print(srctime, f"Loaded fetched traces for {len(event.stations)} stations")
        else: print(f"cannot find event at {filename}")
        return event
    
    def _fetch_par(self, event, generating_event_object=False):
        srctime, starttime, endtime = self._get_srctime(event.srctime)
        filename = f"{self.rawdata_dir}/{srctime}.LH.obspy"
        client = Client("IRIS")

        if not (os.path.exists(filename)):
            print(f"start fetching for the event {srctime}")
            try: fetched_stream = client.get_waveforms("*", "*", "*", "LH?", starttime, endtime, attach_response=True)
            except: print(srctime, '... X (failed fetching)'); return event
            for trace in fetched_stream.select(network="SY"): fetched_stream.remove(trace)
            if len(fetched_stream)>0:
                fetched_stream.write(filename, format="PICKLE")
                
                # sleep or IRIS may cut down your connection
                time.sleep(0.2)
                print(srctime, f"Fetched for {len(event.stations)} stations")
            
            else:
                print(srctime, '... X (no available station exists)')
                
        return len(event.stations)
        

    def fetch(self, cpu_number=None):
        print("setting multiprocessing for fetching...")
        p = ThreadPool(cpu_number or cpu_count()) # set parallel fetch
        t0 = time.time()
        p.map(self._fetch_par, self.events[32020:33872])
        # self.events = p.map(self._fetch_par, self.events[:2000])
        # self.events = p.map(self._fetch_par, self.events[32020:33872]) #only year 2010
        # self.events = p.map(self._fetch_par_spawn, self.events[32020:32056]) #only year 2010
        # self.events = p.starmap(self._prepare_resplist_par, zip(self.events[32020:33872], repeat(False)))
        print(f"fetching finished in {(time.time()-t0):.04f} sec.")
        
    # ### only used in method A for loading station response 
    # def prepare_event_table(self, cpu_number=None):
    #     print("setting multiprocessing for preparing events...")
    #     # p = ThreadPool(cpu_number or cpu_count()) # set parallel fetch
    #     t0 = time.time()
    #     # self.events = p.map(self._prepare_event_table_par, self.events[32020:33872]) #only year 2010
    #     common_args = {'rawdata_dir': self.rawdata_dir, 'resplist': self.resplist}
    #     with get_context('fork').Pool(cpu_number or cpu_count()) as p:
    #     # with ThreadPool(cpu_number or cpu_count()) as p:
    #         self.events = p.starmap(_prepare_event_table_spawn,
    #             zip(self.events[32020:33872], repeat(common_args)))
    #     print(f"event table are prepared in {(time.time()-t0):.04f} sec.")

    # def create_stalist(self, cpu_number=None, respdir='./resp_catalog'): #WIP
    #     loaded_station_count = 0
    #     event_count = 0
    #     if not self.resplist:
    #         print("load station infomation...")
    #         self.prepare_resplist(respdir); self.respdir = respdir
        
    #     for ev in self.events:
    #         if ev: self.numdownloaded+=len(ev.stations); event_count+=1
    #         else: self.events.remove(ev)
    #     print(f"{event_count} events with {self.numdownloaded} seismograms are processed.")

    #     client = Client('IRIS')
    #     obspy_filenames = glob.glob("./rawdata_catalog2/*.obspy")
    #     stalist = {} #{str: list}

    #     for z_trace in stream.select(channel="LHZ"):
    #         station_name = f"{z_trace.stats.network}.{z_trace.stats.station}.LH"
    #         if station_name in stalist:
    #             stalist[station_name].append(datetime_str)
    #         else:
    #             stalist[station_name] = [datetime_str]

    #     # print(f"{len(obspy_filenames)} events are found")
    #     # p = multiprocessing.get_context("fork").Pool(12)
    #     # p.map(func, list(range(len(obspy_filenames[:20]))))
    #     # for i in range(len(obspy_filenames)):
    #     #     func(i)
    #     # print(stalist)
    #     with open('stalist.pkl', 'wb') as file:
    #         pickle.dump(stalist, file)
    #     return
    # ###

    def link_downloaded(self, israwdata=True):
        count = 0
        for event in self.events:
            srctime = event.srctime
            srctime.precision = 3
            savedir = f"{data_rootdir}/rawdata/{srctime}" if israwdata else f"{data_rootdir}/training/{srctime}" 
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
    
    def _prepare_resplist_par(self, item, respdir):
        print(f"preparing instrument {str(item[0])}")
        return (item[0], Instrument(station_str=item[0], event_records=item[1], respdir=respdir))
    
    def prepare_resplist(self, respdir='./resp_catalog', overwrite=False):
        if overwrite or (not os.path.exists(self.response_list_path)):
            with open(self.station_list_path, 'rb') as infile:
                stalist = pickle.load(infile)
                if not 'SR.GRFO.LH' in stalist: stalist['SR.GRFO.LH'] = ['1983-01-01T05:32:01']
                print("station list loaded.")
                with ThreadPool(cpu_count()) as p:
                    self.resplist = dict(p.starmap(self._prepare_resplist_par, zip(stalist.items(), repeat(respdir))))
                with open(self.response_list_path, 'wb') as outfile:
                    pickle.dump(self.resplist, outfile)
        else:
            with open(self.response_list_path, 'rb') as file:
                self.resplist = pickle.load(file)
            print("response list loaded.")

    def _get_datalist_noread(self, event, stream_org, args) -> list:
        obsfile = args['obsfile']
        preprocess = args['preprocess']
        resample = args['resample']
        shift = args['shift']
        rotate = args['rotate']
        reference_responses = args['reference_responses']
        loaddir = args['loaddir']
        savedir = args['savedir']
        overwrite_event = args['overwrite_event']
        resplist = args['resplist']
        velocity_model = args['velocity_model']
        default_p_calctime = args['default_p_calctime']
        event.srctime = UTCDateTime(event.srctime, precision=3)

        sublist = []
        first_writing = False

        def find_location(obspy_obj, station_code=None, location_priority = ['', '00', '10']) -> tuple:
            for location_matching in location_priority:
                if station_code == None:
                    traces_matching = obspy_obj.select(location=location_matching)
                else:
                    traces_matching = obspy_obj.select(station=station_code, location=location_matching)

                if len(traces_matching) == 3: return (location_matching, traces_matching)
                elif len(traces_matching) > 3:
                    if len(traces_matching.select(channel='LH[EN]'))>0:
                        if len(traces_matching.select(channel='LH[ENZ]')) == 3:
                            return (location_matching, traces_matching.select(channel='LH[ENZ]'))
                    else:
                        if len(traces_matching.select(channel='LH[12Z]')) == 3:
                            return (location_matching, traces_matching.select(channel='LH[12Z]'))
            return (None, None)

        # print(f"preparing for {len(event.stations)} stations at {event.srctime}...")

        # load obsfile first if input is compiled
        # if obsfile == 'compiled':
        #     t0 = time.time()
        #     stream_org = read(f"{loaddir}/{event.srctime}.LH.obspy")
        #     print(f"finish loading {event.srctime} in {(time.time()-t0):.04f} sec, now processing...")
        if obsfile != 'compiled' and obsfile != 'mass':
            raise Exception("obsfile must be `compiled` or `mass` in noread method")
        
        event_time_wild = f"{str(UTCDateTime(event.srctime, precision=3))[:-1]}*Z"
        xml_directory = f"{loaddir}/{event.srctime.year}/{event_time_wild}/stations"
        if len(event.stations) == 0:
            print(f"loading station xml files for {event.srctime} at {xml_directory}...")
            for stationxml in glob.glob(f"{xml_directory}/*.xml"):
                try:
                    network = read_inventory(stationxml)[0]; station = network[0]
                except:
                    print(f"failed to read {stationxml}")
                    continue
                # build station list
                location_matched, _ = find_location(station)
                if location_matched == None: continue
                fetched = Station(station.code, station.latitude, station.longitude,
                                    dist=locations2degrees(lat1=station.latitude, long1=station.longitude, lat2=event.srcloc[0], long2=event.srcloc[1]),
                                    azi=gps2dist_azimuth(lat1=station.latitude, lon1=station.longitude, lat2=event.srcloc[0], lon2=event.srcloc[1])[2],
                                    loc=location_matched)
                fetched.labelnet['code'] = network.code
                fetched.isdataexist = True
                event.stations.append(fetched)

        # loop for trace
        for trace_set in event.stations:
            network_code = trace_set.labelnet['code']
            station_code = trace_set.labelsta['name']
            station_label = trace_set.labelsta
            event_name = f"{station_code}.{network_code}_{event.srctime.year:4d}{event.srctime.month:02d}{event.srctime.day:02d}{event.srctime.hour:02d}{event.srctime.minute:02d}{event.srctime.second:02d}_EV"
            processed_filename = f"{savedir}/{event.srctime.year:4d}/{event.srctime}/{network_code}.{station_code}.LH.obspy"

            # network filter
            # if not network_code in ['AU', 'BR', 'DK', 'G', 'GE', 'GT', 'II', 'IM', 'IU', 'PS', 'SR']: continue

            # load proceesed file if exists
            if not os.path.exists(f"{savedir}/{event.srctime.year:4d}"): os.mkdir(f"{savedir}/{event.srctime.year:4d}")
            if not overwrite_event and os.path.exists(processed_filename):
                stream = read(processed_filename)
                #print(f"loaded {event_name}")
                
            # prepare proprocessing if not
            else: 
                # continue ## adhoc only 400
                # load obsfile now if waveforms are mass downloaded mseed
                if obsfile == 'mass':
                    event_time_wild = f"{str(UTCDateTime(event.srctime, precision=3))[:-1]}*Z"
                    if network_code == None:
                        network_search = f"{loaddir}/{event.srctime.year}/{event_time_wild}/waveforms/*.{station_code}.*.mseed"
                        if len(glob.glob(network_search)) == 0:
                            print(f"missing event:", network_search); continue
                        else:
                            network_code = network_search[0].split('/')[-1].split('.')[0]
                            event_name = f"{station_code}.{network_code}_{event.srctime.year:4d}{event.srctime.month:02d}{event.srctime.day:02d}{event.srctime.hour:02d}{event.srctime.minute:02d}{event.srctime.second:02d}_EV"
                            processed_filename = f"{savedir}/{event.srctime.year:4d}/{event.srctime}/{network_code}.{station_code}.LH.obspy"

                    station_search = f"{loaddir}/{event.srctime.year}/{event_time_wild}/waveforms/{network_code}.{station_code}.*.mseed"
                    if not glob.glob(station_search):
                        print(f"missing waveforms:", station_search); continue
                    try: stream_org = read(station_search)
                    except InternalMSEEDError as e:
                        print(f"Corrputed mseed in {event_name}: {e}, trying to read individually...") 
                        stream_org = Stream()
                        for filename in glob.glob(station_search):
                            try: stream_org.append(read(filename)[0])
                            except: pass
                # select 3 components
                if ('loc' in station_label) and (station_label['loc'] is not None):
                    if type(station_label['loc'] is list): # bug-handling
                        _, stream = find_location(stream_org, station_code)
                        if stream is None: continue
                    else:
                        stream = stream_org.select(station=station_code, location=station_label['loc'])
                        if len(stream) > 3:
                            if len(stream.select(channel='LHE'))>0 and len(stream.select(channel='LHN'))>0:
                                stream.remove(stream.select(channel='LH1'))
                                stream.remove(stream.select(channel='LH2'))
                            else:
                                stream.remove(stream.select(channel='LHN'))
                                stream.remove(stream.select(channel='LHE'))
                        stream.sort()
                else:
                    location_priority = ['', '00', None]
                    for location_matching in location_priority:
                        traces_matching = stream_org.select(station=station_code, location=location_matching)
                        if len(traces_matching) == 3: break
                        elif len(traces_matching) > 3:
                            try:
                                if len(traces_matching.select(channel='LH[ENZ]')) == 3:
                                    if len(traces_matching.select(channel='LH[12]'))>0: traces_matching.remove(list(traces_matching.select(channel='LH[12]')))
                                    if len(traces_matching) == 3: break
                                elif len(traces_matching.select(channel='LH[12Z]')) == 3:
                                    if len(traces_matching.select(channel='LH[EN]'))>0: traces_matching.remove(list(traces_matching.select(channel='LH[EN]')))
                                    if len(traces_matching) == 3: break
                            except ValueError as e: # idk sometimes not-in-list error happens when remove
                                print(f"Value error for {event_name}: {e}")
                    if location_matching == None: continue
                    stream = traces_matching

                # sanity check for all three component
                if not (len(stream) == 3 and len(stream[0])*len(stream[1])*len(stream[2])>0 and np.isscalar(stream[0].data[0])): continue

                # preprocessing
                #print(f"preprocessing {event_name}...")
                try:
                    # check array size for waveform data
                    stream.trim(starttime=fn_starttime_full(event.srctime), endtime=fn_endtime_full(event.srctime)+1)
                    for record in stream:
                        if record.data.shape[0] < 9000: raise ValueError(f"Trace has too few samples: {str(record)}")
                        elif record.data.shape[0] > 9000+20: raise ValueError(f"Trace has too many samples: {str(record)}")
                        elif record.data.shape[0] <= 9000+20 and record.data.shape[0] > 9000: record.data = record.data[:9000]

                    # deconvolve and convolve instrument response using obspy
                    if preprocess is True:
                        if obsfile == 'mass': # look for station xml
                            obspy_station = read_inventory(f'{xml_directory}/{network_code}.{station_code}.xml')[0][0]
                        else: # read from resplist
                            obspy_station = resplist[f'{network_code}.{station_code}.LH'].find_obspy_station(event.srctime)
                        if len(stream.select(channel="LHE"))>0 and len(stream.select(channel="LHN"))>0:
                            channels = ["LHE", "LHN", "LHZ"]
                            stream = deconv_resp(rawdata=stream, 
                                                    station_resps=[obspy_station.select(channel=channel,time=event.srctime)[0].response for channel in channels],
                                                    reference_resps=reference_responses)
                        else:
                            channels = ["LH2", "LH1", "LHZ"]
                            stream = deconv_resp(rawdata=stream, 
                                                    station_resps=[obspy_station.select(channel=channel,time=event.srctime)[0].response for channel in channels],
                                                    reference_resps=reference_responses)
                            stream[0].meta.channel = "LHE"
                            stream[1].meta.channel = "LHN"

                    # rotate to TRZ coordinate
                    if rotate:
                        #load or calculate azimuth angle
                        azimuth = gps2dist_azimuth(lat1=station_label['lat'], lon1=station_label['lon'], lat2=event.srcloc[0], lon2=event.srcloc[1])[2] if station_label['azi'] is None else station_label['azi']
                        stream.rotate('NE->RT', back_azimuth=azimuth)
                    
                    # bandpass filter
                    if preprocess == 'bandpass':
                        stream.filter('bandpass', freqmin=0.03, freqmax=0.05, corners=2, zerophase=True)

                    # save data
                    if not os.path.exists(f"{savedir}/{event.srctime.year:04d}/{event.srctime}"):
                        try: os.makedirs(f"{savedir}/{event.srctime.year:04d}/{event.srctime}")
                        except FileExistsError: pass # sometimes happens when folder created by another subprocess
                        except: raise FileNotFoundError(f"error when opening a new folder: {savedir}/{event.srctime.year:04d}/{event.srctime}")
                    if not first_writing: print(f"now writing first trace for {network_code}.{station_code} for event {event.srctime}"); first_writing = True
                    stream.write(processed_filename, format="PICKLE")
                    
                except IndexError as e:
                    print(f"Index error for {event_name}: {e}")
                except ValueError as e:
                    print(f"Value error for {event_name}: {e}")
                except FileNotFoundError as e:
                    print(f"File-not-found error for {event_name}: {e}")
                except ObsPyException as e:
                    print(f"ObsPy error for {event_name}: {e}")

            # prepare shifting and labeling
            try:
                # load p info
                p_status = None
                p_weight = None
                p_travel_sec = None
                is_calctim_defined = False

                for record in trace_set.records:
                    if record.phase == 'P':
                        p_status = 'manual'
                        p_weight = 1/record.error
                        p_calctim0 = event.srctime + record.calctim
                        p_obstim0 = event.srctime + record.obstim
                        is_calctim_defined = True
                
                if not is_calctim_defined:
                    try:
                        p_calctim0 = event.srctime + velocity_model.get_travel_times(event.srcloc[2], station_label['dist'], ['P'])[0].time
                        is_calctim_defined = True
                    except: p_calctim0 = event.srctime + default_p_calctime

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

                    for record in trace_set.records:
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
                    ### should add theroteical arrival time
                    waveform_attrs = {
                        'network_code': network_code,
                        'receiver_code': station_code,
                        'receiver_type': 'LH',
                        'receiver_latitude': station_label['lat'], 
                        'receiver_longitude': station_label['lon'], 
                        'receiver_elevation_m': None, 
                        'p_arrival_sample': int(p_travel_sec/delta) if p_status else None, 
                        'p_status': p_status, 
                        'p_weight': p_weight, 
                        'p_travel_sec': p_travel_sec, 
                        'p_calculate_travel_sec': p_calctim0 - event.srctime if is_calctim_defined else None, 
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
                        'source_distance_deg': station_label['dist'], 
                        'source_distance_km': station_label['dist'] * 111.1, 
                        'back_azimuth_deg': station_label['azi'], 
                        'snr_db': snr, 
                        'coda_end_sample': coda_end_sample, 
                        'trace_start_time': str(event.srctime), 
                        'trace_category': 'earthquake_local', 
                        'trace_name': event_name
                        }
                    
                    sublist.append({'data': waveform_data, 'attrs': waveform_attrs})

            except UnboundLocalError as e:
                print(f"Unbound local error for {event_name}: {e}")
            except ValueError as e:
                print(f"Value error for {event_name}: {e}")
            except FileNotFoundError as e:
                print(f"File-not-found error for {event_name}: {e}")
        
        print(f"finished preparing for {len(event.stations)} stations at {event.srctime}.")
        return sublist
        
    # create datalist by searching all records in the event table
    def _datalist_while_reading_obspy(self, events, common_args):
        results = [None] * len(events)
        loaddir = common_args['loaddir']
        event_already_read = events[0]
        t0 = time.time()
        stream_org = read(f"{loaddir}/{event_already_read.srctime}.LH.obspy")
        print(f"finish loading {event_already_read.srctime} in {(time.time()-t0):.04f} sec, now processing...")

        def read_and_replace(event_to_read):
            t0 = time.time()
            stream_org = read(f"{loaddir}/{event_to_read.srctime}.LH.obspy")
            print(f"finish loading {event_to_read.srctime} in {(time.time()-t0):.04f} sec, now processing...")
            return stream_org
        
        for index in range(len(events)-1):
            event_to_read = events[index+1]
            with ThreadPoolExecutor(max_workers=2) as executor:
                thread1 = executor.submit(self._get_datalist_noread, event_already_read, stream_org, common_args)
                thread2 = executor.submit(read_and_replace, event_to_read)

            results[index] = thread1.result()
            stream_org = thread2.result()
            event_already_read = event_to_read
        
        results[-1] = self._get_datalist_noread(event_already_read, stream_org, common_args)
        return results
    
    # create datalist by searching all records in the event table
    def _datalist_for_mass(self, events, common_args):
        return [self._get_datalist_noread(event, None, common_args) for event in events]

    def get_datalist(self, resample=0, rotate=True, preprocess=True, shift=(-100,100), output='./test.hdf5', overwrite_hdf=True, overwrite_event=False, obsfile='separate', year_option=None, cutoff_magnitude=None, dir_ext='', cpu_number=None, batch_size=20, respdir='./resp_catalog'):
        if not shift: shift = (0,0)
        
        # load station list and build inventory for station response
        if preprocess:
            if obsfile == 'mass':
                reference_inventory = read_inventory(f"{respdir}/SR.GRFO.1978.225.00.1993.329.00")[0][0]
                reference_responses = [reference_inventory.select(channel=channel)[0].response for channel in ["LHE", "LHN", "LHZ"]]

            else:
                if not self.resplist: self.prepare_resplist(respdir)
                # get instrument response for reference station
                # reference_responses = [InstrumentResponse(network='SR', station='GRFO', component=component, timestamp=UTCDateTime(1983, 1, 1)) for component in ['LHE', 'LHN', 'LHZ']]
                reference_datetime = UTCDateTime(1983, 1, 1)
                reference_responses = [self.resplist['SR.GRFO.LH'].find_obspy_station(reference_datetime).select(channel=channel,time=reference_datetime)[0].response for channel in ["LHE", "LHN", "LHZ"]]
            print("reference station response file is read")

        # set directory
        loaddir = f'{data_rootdir}/rawdata{dir_ext}' if preprocess else f"{data_rootdir}/training{dir_ext}"
        if preprocess=='bandpass': savedir = f"{data_rootdir}/training_bandpass{dir_ext}"
        elif preprocess=='onlyrot': savedir = f"{data_rootdir}/training_onlyrot{dir_ext}"
        elif preprocess: savedir = f"{data_rootdir}/training{dir_ext}"

        # select events
        def event_checked(event) -> bool:
            return False if (year_option and (event.srctime.year != year_option)) or (cutoff_magnitude and (event.magnitude < cutoff_magnitude)) else True
        self.events = list(filter(event_checked, self.events))
        print("events to process are selected")

        # set up parameters for parallel computing
        common_args = {
            'obsfile': obsfile,
            'preprocess': preprocess,
            'resample': resample,
            'shift': shift,
            'rotate': rotate,
            'reference_responses': reference_responses,
            'loaddir': loaddir,
            'savedir': savedir,
            'overwrite_event': overwrite_event,
            'resplist': self.resplist if obsfile != 'mass' else None,
            'velocity_model': self.picker.model,
            'default_p_calctime': self.picker.default_p_calctime
        }

        # # create datalist by searching all records in the event table
        # datalist = []
        # print(f"preparing waveforms in parallel, #cpu={cpu_number or cpu_count()}.")
        # t0 = time.time()
        # with ThreadPool(cpu_number or cpu_count()) as p: # event chucks
        #     batch_results = p.starmap(_get_datalist_par,
        #                               zip(self.events[:10], repeat(common_args)))
        # # batch_results = [_get_datalist_par(event, common_args) for event in self.events[:48]]
        # print(f"test run finished in {(time.time()-t0):.04f} sec")

        
        worker_number = (cpu_number or cpu_count())
        shuffled_events = picker.data.events.copy()
        random.shuffle(shuffled_events)

        with h5py.File(output, 'w' if overwrite_hdf else 'a') as f:
            waveform_count = 0
            datalist = []
            if overwrite_hdf: f.create_group("data")

            while (shuffled_events):
                popped_events = [shuffled_events.pop() for _ in range(min(worker_number*batch_size, len(shuffled_events)))]
                print(f"preparing waveform batches in parallel, #worker={worker_number}.")
                t0 = time.time()
                with ProcessPoolExecutor(max_workers=worker_number) as executor:
                    threads = [None] * (worker_number)
                    input_pool = split_array(popped_events, worker_number)
                    for thread_index in range(worker_number):
                        thread_input = input_pool[thread_index]
                        if obsfile == 'mass':
                            threads[thread_index] = executor.submit(self._datalist_for_mass, thread_input, common_args)
                        else:
                            threads[thread_index] = executor.submit(self._datalist_while_reading_obspy, thread_input, common_args)

                batch_results = [threads[thread_index].result() for thread_index in range(worker_number)]

                # batch_results = [self._datalist_for_mass(shuffled_events, common_args)]
                print(f"batch run finished in {(time.time()-t0):.04f} sec")

                # writing part
                flatten_results = flatten_list(batch_results)
                for item in flatten_results:
                    if f"data/{item['attrs']['trace_name']}" in f: continue
                    dataset = f.create_dataset(f"data/{item['attrs']['trace_name']}", data=item['data'])
                    for attr, val in item['attrs'].items():
                        if not val is None: dataset.attrs[attr] = val
                    item['attrs']['source_origin_time'] = UTCDateTime(item['attrs']['source_origin_time'])
                    item['attrs']['trace_start_time'] = UTCDateTime(item['attrs']['trace_start_time'])
                    item['attrs']['coda_end_sample'] = [[item['attrs']['coda_end_sample']]]
                    datalist.append(item['attrs'])
                    waveform_count = waveform_count + 1

        print(f"All {waveform_count} waveforms from {len(self.events)} events by {self.events[-1].srctime} are done.")
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


    def __init__(self, picker, client: Client, paths: list, dataset_as_folder=False, **kwargs):
        self.picker = picker
        self.client = client
        self.numdownloaded = 0
        self.resplist = None
        self.working_freqencies = None
        autofetch = kwargs['autofetch'] if 'autofetch' in kwargs else False
        self.station_list_path = kwargs['station_list_path'] if 'station_list_path' in kwargs else "./stalist.pkl"
        self.response_list_path = kwargs['response_list_path'] if 'response_list_path' in kwargs else "./resp_catalog/resplist.pkl"
        self.rawdata_dir = kwargs['rawdata_dir'] if 'rawdata_dir' in kwargs else "./rawdata_catalog3"

        # create event list from paths
        self.events = []
        for path in paths:
            if dataset_as_folder:
                numevent = self._folder2events(path)
                print(f"folder read successfully, {numevent} events added")
            else:
                numevent = self._table2events(path)
                print(f"table read successfully, {numevent} events added")

        # fetch data if autofetch toggled
        if autofetch: self.fetch()
    

class Picker():
    def __init__(self, dataset_paths, dataset_as_folder=False, default_p_calctime=450, stationlist_method="event_table", target_year=None, preproc_csv_filename=None, online_mode=False, **kwargs):
        self.client = Client('IRIS') if online_mode is True else None
        self.model = TauPyModel(model="prem")
        self.station_list = None
        self.station_dict = None
        self.stationlist_method = stationlist_method
        self.target_year = target_year
        self.default_p_calctime = default_p_calctime
        self.preproc_csv_filename = preproc_csv_filename
        self.create_dataset(dataset_paths, dataset_as_folder, **kwargs)

    def create_dataset(self, table_filenames, dataset_as_folder, **kwargs):
        "create dataset from scretch"
        self.data = SeismicData(self, self.client, table_filenames, dataset_as_folder, **kwargs)

    def load_dataset(self, filename, verbose=False, magnitude_ref=None):
        "load existing dataset"
        with open(filename, 'rb') as file:
            loaddata = pickle.load(file)
        # self.data = SeismicData(self, self.client, [])
        self.data.events = loaddata.events
        if magnitude_ref:
            if type(magnitude_ref) is str: magnitude_ref = np.load(magnitude_ref, allow_pickle=True)
            for event in self.data.events:
                matched_event = None
                for searched_event in magnitude_ref:
                    if searched_event == event: matched_event = searched_event; break
                event.magnitude = matched_event.magnitude

        # debug message
        if verbose: 
            printphase = [record.phase for record in self.data.events[0].stations[3].records]
            print(f"loaded dataset has {len(self.data.events)} events")
            print(f"#1 event has {len(self.data.events[0])} stations, records of #4 event: {', '.join(printphase)}")
    
    def dump_dataset(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.data, file, pickle.HIGHEST_PROTOCOL)

    def get_stationlist(self):
        if self.stationlist_method == "event_table":
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
        else:
            station_list = []
            station_dict = {}
            for event in self.data.events:
                if event.srctime.year != self.target_year: continue
                event_time_wild = f"{str(UTCDateTime(event.srctime, precision=3))[:-1]}*Z"
                search_path = f"{self.data.rawdata_dir}/{event.srctime.year}/{event_time_wild}/stations/*.xml"
                found_path = glob.glob(search_path)
                # print(f"found {len(found_path)} station files")
                for filename in found_path:
                    station_code = filename.split("/")[-1].split(".")[1] 
                    if not station_code in station_list:
                        station_list.append(station_code)
                        inv = read_inventory(filename)
                        station_dict[station_code] = Station(station_code, inv[0][0].latitude, inv[0][0].longitude,
                                    dist=None, azi=None, loc=None)
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
    def _prepare_picking_par(self, station_code, overlap=0, obsfile='mass'):
        output_name = station_code
        station = self.station_dict[station_code]
        
        try:
            os.remove(f'{self.save_dir}/{output_name}.hdf5')
            os.remove(f'{self.save_dir}/{output_name}.csv')
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
        elif obsfile=='mass':
            # filenames = glob.glob(f'{self.waveform_dir}/*/*.{station_code}.*')
            df = pd.read_csv(self.preproc_csv_filename)
            filtered_df = df[df['receiver_code'] == station_code]
            filenames = [ f"{self.waveform_dir}/{str(UTCDateTime(row['source_origin_time'], precision=3))}/{row['network_code']}.{station_code}.LH.obspy" for index, row in filtered_df.iterrows() ]
        else:
            filenames = glob.glob(f'{self.waveform_dir}/*.{station_code}.*')
            # filenames = glob.glob(f'{self.waveform_dir}/2010*.{station_code}.*')
        
        time_slots, comp_types = [], []
        csv_references = {}
        
        if len(filenames) == 0: return
        print('============ Station {} has {} chunks of data.'.format(station_code, len(filenames)), flush=True)  
            
        count_chuncks=0; c1=0; c2=0; c3=0
        
        for filename in filenames:
            try:
                st = read(filename, debug_headers=True)
            except:
                print(f"failed to open: {filename}")
                continue
            component_num = len(st)

            if obsfile=='separate' or obsfile=='mass':
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
                    
                    if not target_srctime.year in csv_references:
                        if os.path.exists(f"{result_rootdir}/{preproc_prefix}_{target_srctime.year}_preproc.csv"):
                            csv_references[target_srctime.year] = pd.read_csv(f"{result_rootdir}/{preproc_prefix}_{target_srctime.year}_preproc.csv")
                        else:
                            csv_references[target_srctime.year] = None
                    ref_csv = csv_references[target_srctime.year]

                    if (ref_csv is not None) and ('p_calculate_travel_sec' in ref_csv.columns):
                        records_matched = ref_csv.loc[(ref_csv['source_origin_time'] == str(UTCDateTime(target_srctime, precision=6))) & (ref_csv['receiver_code'] == station_code)]
                        if len(records_matched) > 0:
                            if records_matched['p_calculate_travel_sec'].iloc[0] > 0:
                                p_calctim = records_matched['p_calculate_travel_sec'].iloc[0]
                                print("Using calculated P arrival window for", output_name, filename)
                            else: print("Using default P arrival window for", output_name, filename)
                        else: print("Using default P arrival window for", output_name, filename)
                            
                    else:
                    # if True:
                        for event in self.data.events:
                            if event.srctime == target_srctime:
                                ref_trace = event._findstation(station)
                                if obsfile == 'separate':
                                    ref_trace = event._findstation(station)
                                    dist = ref_trace.labelsta['dist']
                                else: # i.e., if obsfile == 'mass'
                                    event_time_wild = f"{str(UTCDateTime(event.srctime, precision=3))[:-1]}*Z"
                                    sta = read_inventory(f"{self.data.rawdata_dir}/{event.srctime.year}/{event_time_wild}/stations/*{station_code}.xml")[0][0]
                                    dist = locations2degrees(lat1=sta.latitude, long1=sta.longitude, lat2=event.srcloc[0], long2=event.srcloc[1])
                                try:
                                    p_calctim = self.model.get_travel_times(event.srcloc[2], dist, ['P'])[0].time
                                    # p_calctim = self.model.get_travel_times(event.srcloc[2], ref_trace.labelsta['dist'], ['P'])[0].time
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
                            if obsfile == 'compiled':
                                ref_trace = event._findstation(station)
                                dist = ref_trace.labelsta['dist']
                            else:
                                event_time_wild = f"{str(UTCDateTime(event.srctime, precision=3))[:-1]}*Z"
                                sta = read_inventory(f"{self.data.rawdata_dir}/{event.srctime.year}/{event_time_wild}/stations/*{station_code}.xml")[0][0]
                                dist = locations2degrees(lat1=sta.latitude, long1=sta.longitude, lat2=event.srcloc[0], long2=event.srcloc[1])
                            print(p_calctim = self.model.get_travel_times(event.srcloc[2], dist, ['P'])[0].time)
                            try:
                                p_calctim = self.model.get_travel_times(event.srcloc[2], dist, ['P'])[0].time
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
            print("generating station list...")
            self.get_stationlist()

        self.model = TauPyModel(model="prem")
        self.data_track = dict()

        # print(len(self.station_list))

        if not n_processor:
            n_processor = cpu_count()
        with ThreadPool(n_processor) as p:
            p.map(self._prepare_picking_par, self.station_list) 
        with open(os.path.join(preproc_dir,'time_tracks.pkl'), 'wb') as f:
            pickle.dump(self.data_track, f, pickle.HIGHEST_PROTOCOL)


# if __name__ == '__main__':
#     # table_dir = "../Downloads/drive-download-20220512T014633Z-001"
#     # table_filenames = [f"{table_dir}/Pcomb.4.07.09.table",f"{table_dir}/Scomb.4.07.09.table"]
#     # picker = Picker(table_filenames, False,
#     #         station_list_path=None,#"./stalist2010.pkl",
#     #         response_list_path=None,#"./resp_catalog/resplist2010_lite.pkl",
#     #         rawdata_dir="./rawdata_catalog_mass",
#     #         # target_year=int(sys.argv[1]),
#     #         stationlist_method="directory"
#     #         )
#     # # picker.create_dataset(table_filenames)
#     # np.save('HMSL_labeled_evnets', picker.data.events)
#     picker = Picker([], False,
#             station_list_path=None,#"./stalist2010.pkl",
#             response_list_path=None,#"./resp_catalog/resplist2010_lite.pkl",
#             rawdata_dir="./rawdata_catalog_mass",
#             # target_year=int(sys.argv[1]),
#             stationlist_method="directory"
#             )
#     print("picker created.")
#     picker.load_dataset('data.pkl', verbose=True)
#     # print(len(picker.data.events), "events")
#     np.save('HMSL_labeled_evnets', picker.data.events)

if __name__ == '__main__':
    data_rootdir = '.'
    datafolder_ext = "_catalog_mass_not_usta" 
    preproc_prefix = "catalog_not_usta"
    result_rootdir = '.'
    table_dir = './drive-download-20220512T014633Z-001'
    gcmt_catalog_dir = '../waveformget/gcmt_mw.npy'
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

    # best workflow:
    if not (int(sys.argv[1]) > 1970 and int(sys.argv[1]) < 2100): raise ValueError("invalid args")
    preproc_csv_filename = f'{result_rootdir}/{preproc_prefix}_{int(sys.argv[1])}_preproc.csv'
    preproc_hdf_filename = f'{result_rootdir}/{preproc_prefix}_{int(sys.argv[1])}_preproc.hdf5'
    picker = Picker([], False,
            station_list_path=None,#"./stalist2010.pkl",
            response_list_path=None,#"./resp_catalog/resplist2010_lite.pkl",
            rawdata_dir=os.path.join(data_rootdir, f"rawdata{datafolder_ext}"),
            target_year=int(sys.argv[1]),
            preproc_csv_filename = preproc_csv_filename,
            # rawdata_dir="./rawdata_HMSL_labeled",
            stationlist_method="directory"
            )
    print("picker created.")

    picker.data.events = list(np.load(gcmt_catalog_dir, allow_pickle=True))
    # picker.data.events = list(np.load('./HMSL_labeled_evnets.npy', allow_pickle=True))
    print("catalog loaded.")

    ################################################################
    # A: download events by specifying a time frame
    # -> simply download all GCMT cataloged LH data (prefered # of MP downloading sessions is 10? for IRIS) by data.fetch
    # picker.data.fetch(cpu_number=10)

    # -> make station list into stalist.pkl by `python stalist.py`

    # # -> sort response list into resplist.pkl by data.prepare_resplist()
    # picker.data.prepare_resplist(respdir='./resp_catalog', overwrite=True)

    # # -> read the final resplist.pkl to generate event-station datalist into data_fetched_catalog.pkl by data.prepare_event_table()
    # picker.data.prepare_resplist(respdir='./resp_catalog')
    # picker.data.prepare_event_table(cpu_number=12)
    # picker.dump_dataset("./rawdata_catalog3/data_fetched_catalog_2010_3.pkl")

    # # -> preproc the datalist into training_catalog/* and catalog_preproc.hdf5 by data.get_datalist()
    # # picker.data.prepare_resplist(respdir='./resp_catalog')
    # picker.load_dataset('./rawdata_catalog3/data_fetched_catalog_2010_3.pkl', verbose=True)
    # print("station records loaded.")
    # datalist = picker.data.get_datalist(resample=resample_rate, preprocess=True, output='./rawdata_catalog3/catalog_2010_preproc_3_re.hdf5', overwrite_hdf=True, obsfile="compiled", year_option=2010, cutoff_magnitude=5.5, dir_ext='_catalog3', cpu_number=8)
    # df = pd.DataFrame(datalist)
    # df.to_csv('catalog_2010_preproc_3.csv', index=False)
    ################################################################
    
    ################################################################
    # B: download events with mass downloader
    # call mass downloader 'waveformget'

    # -> preproc the datalist into training_catalog/* and catalog_preproc.hdf5 by data.get_datalist()
    # datalist = picker.data.get_datalist(
    #     resample=resample_rate,
    #     preprocess=True,
    #     year_option=picker.target_year,
    #     cutoff_magnitude=5.5,
    #     obsfile="mass",
    #     dir_ext='_catalog_mass',
    #     overwrite_hdf=True,
    #     output=f'./catalog_{picker.target_year}_preproc.hdf5',
    #     cpu_number=10
    # )
    # df = pd.DataFrame(datalist)
    # df.to_csv(f'catalog_{picker.target_year}_preproc.csv', index=False)    

    # datalist = picker.data.get_datalist(
    #     resample=resample_rate,
    #     preprocess=True,
    #     # year_option=picker.target_year,
    #     # cutoff_magnitude=5.5,
    #     obsfile="mass",
    #     dir_ext='_HMSL_labeled',
    #     overwrite_hdf=True,
    #     overwrite_event=True,
    #     output=f'./HMSL_labeled_preproc.hdf5',
    #     respdir='/Volumes/seismic/resp_catalog',
    #     cpu_number=10
    # )
    # df = pd.DataFrame(datalist)
    # df.to_csv(f'HMSL_labeled_preproc.csv', index=False)    

    datalist = picker.data.get_datalist(
        resample=resample_rate,
        preprocess=True,
        year_option=picker.target_year,
        cutoff_magnitude=5.5,
        obsfile="mass",
        dir_ext=datafolder_ext,
        overwrite_hdf=True,
        output=preproc_hdf_filename,
        cpu_number=10,
        batch_size=10
    )
    df = pd.DataFrame(datalist)
    df.to_csv(preproc_csv_filename, index=False)

    ################################################################

    # -> prepare directory for prediction by picker.prepare_catalog()
    # preproc_prefix = "catalog_not_usta"
    # picker.load_dataset('{result_rootdir}/rawdata_catalog3/data_fetched_catalog_2010_3.pkl', verbose=True)
    # picker.prepare_catalog(f'{result_rootdir}/training_HMSL_labeled/*', f'/Volumes/seismic/training_HMSL_labeled_preproc', f'/Volumes/seismic/training_HMSL_labeled_hdfs', 10)
    # picker.prepare_catalog(f'{result_rootdir}/training_catalog_mass/{picker.target_year}', f'/Volumes/seismic/catalog_{picker.target_year}_stnflt_preproc', f'/Volumes/seismic/catalog_{picker.target_year}_stnflt_hdfs', 10)
    # picker.prepare_catalog(f'{result_rootdir}/training_catalog_mass_not_usta/{picker.target_year}', f'/Volumes/seismic/catalog_not_usta_{picker.target_year}_stnflt_preproc', f'/Volumes/seismic/catalog_not_usta_{picker.target_year}_stnflt_hdfs', 10)
    # picker.prepare_catalog(f'/Volumes/SSD-PGMU3/training_catalog_mass_not_usta/{picker.target_year}', f'/Volumes/SSD-PGMU3/{preproc_prefix}_{picker.target_year}_stnflt_preproc', f'/Volumes/SSD-PGMU3/{preproc_prefix}_{picker.target_year}_stnflt_hdfs', 10)
    picker.prepare_catalog(f'{data_rootdir}/training{datafolder_ext}/{picker.target_year}', f'{result_rootdir}/{preproc_prefix}_{picker.target_year}_preproc', f'{result_rootdir}/{preproc_prefix}_{picker.target_year}_hdfs', 10)
    # -> run predition with EQTransfomer in JupyterNotebook

    # # create dataset from scretch, fetch seismic data, and dump
    # picker = Picker()
    # picker.create_dataset([])
    # catalog = np.load('./gcmt.npy',allow_pickle=True)
    # picker.data.events = catalog
    # print("catalog loaded.")
    # picker.data.fetch(cpu_number=10)
    # # picker.dump_dataset("./rawdata_catalog2/data_fetched_catalog_2010_2.pkl")

    # # load fetched dataset, remove instrument response, and create training dataset
    # # picker = Picker()
    # # picker.load_dataset('./rawdata_catalog2/data_fetched_catalog_2010.pkl', verbose=True)
    # datalist = picker.data.get_datalist(resample=resample_rate, preprocess=True, output='./rawdata_catalog2/catalog_2010_preproc_2.hdf5', overwrite_hdf=True, obsfile="compiled", year_option=2010, dir_ext='_catalog2')
    # df = pd.DataFrame(datalist)
    # df.to_csv('catalog_2010_preproc_2.csv', index=False)
    # # datalist = picker.data.get_datalist(resample=resample_rate, rotate=True, preprocess=False, shift=False, output='./updeANMO.hdf5', obsfile="compiled", year_option=2010, dir_ext='_catalog2')
    # # random.shuffle(datalist)
    # # df = pd.DataFrame(datalist)
    # # df.to_csv('training_PandS_updeANMO.csv', index=False)

    # # # load dataset and prepare prediction data
    # # picker = Picker(default_p_calctime=450)
    # # picker.load_dataset('./rawdata_catalog2/data_fetched_catalog_2010.pkl', verbose=True)
    # picker.prepare_catalog('./training_catalog2', '/Volumes/seismic/catalog_preproc2', '/Volumes/seismic/catalog_hdfs2')
    # # picker.data.prepare_resplist()

    # picker = Picker()
    # picker.load_dataset('data_fetched_catalog.pkl', verbose=False)
    # summ=0
    # for ev in picker.data.events:
    #     summ+=len(ev.stations)
    # print(summ)
