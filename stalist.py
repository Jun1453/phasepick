import glob
import time
import obspy
import pickle
import multiprocessing
from obspy.clients.fdsn import Client

fn_starttime_full = lambda srctime: srctime - 0.5 * 60 * 60
fn_endtime_full = lambda srctime: srctime + 2 * 60 * 60

client = Client('IRIS')
obspy_filenames = glob.glob("./rawdata_catalog2/*.obspy")
stalist = {} #{str: list}

def func(i):
    print(f"procesing #{i}: {obspy_filenames[i]}")
    datetime_str = obspy_filenames[i].split('/')[-1].split('.')[0]
    try:
        stream = obspy.read(obspy_filenames[i])
    except:
        srctime = obspy.UTCDateTime(datetime_str)
        srctime.precision = 3
        starttime = fn_starttime_full(srctime)
        endtime = fn_endtime_full(srctime)
        filename = f"./rawdata_catalog2/{srctime}.LH.obspy"
        client.get_waveforms("*", "*", "*", "LHZ,LHN,LHE", starttime, endtime, attach_response=True, filename=filename)
        try:
            stream = obspy.read(obspy_filenames[i])
        except:
            print(f"ERROR cannot redownload waveforms for #{i}: {obspy_filenames[i]}")
            return

    for z_trace in stream.select(channel="LHZ"):
        station_name = f"{z_trace.stats.network}.{z_trace.stats.station}.LH"
        if station_name in stalist:
            stalist[station_name].append(datetime_str)
        else:
            stalist[station_name] = [datetime_str]
            # time.sleep(0.05*(i+1))
            # if station_name in stalist:
            #     stalist[station_name].append(datetime_str)
            # else:
            #     stalist[station_name] = [datetime_str]
        # traces_selected = stream.select(station=z_trace.stats.station)
        # if len(traces_selected) < 3:
        #     continue
        # # record location if all traces agree
        # elif len(traces_selected) == 3:
        #     if (traces_selected[0].stats.location==traces_selected[1].stats.location) and (traces_selected[1].stats.location==traces_selected[2].stats.location):
        #         # location = z_trace.stats.location
        #         station_name = f"{z_trace.stats.network}.{z_trace.stats.station}.LH.{z_trace.stats.location}"
                
        #     else: continue
        # elif len(traces_selected) > 3:
        #     # reselect for location '' if more than 3 components
        #     traces_selected = fetched_stream.select(station=trace.stats.station, location='')
        #     if len(traces_selected) == 3: location = traces_selected[0].stats.location
        #     # reselect for location '00' if more than 3 components
        #     else:
        #         traces_selected = fetched_stream.select(station=trace.stats.station, location='00')
        #         if len(traces_selected) == 3: location = traces_selected[0].stats.location
        #         else: continue
    
# print(f"{len(obspy_filenames)} events are found")
# p = multiprocessing.get_context("fork").Pool(12)
# p.map(func, list(range(len(obspy_filenames[:20]))))
for i in range(len(obspy_filenames)):
    func(i)
# print(stalist)
with open('stalist.pkl', 'wb') as file:
    pickle.dump(stalist, file)

                
# def func_test(a):
#     return a+2
# p = multiprocessing.get_context("fork").Pool(2)
# res = p.map(func_test, list(range(12)))
# print(res)
