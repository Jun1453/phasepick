
import glob
import numpy as np
import pandas as pd
from cmcrameri import cm
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from export_catalog import plot_depthslice, plot_ratio

gcarc_range = (70,75)
nskip = 1
cmap = cm.vik

p_thres = 0.7
s_thres = 0.5
# plt.style.use('dark_background')

get_p_table = lambda gcarc_range: plot_depthslice(phase = 'p',
                    value = 'anomaly',
                    gcarc_range = gcarc_range,
                    fidelity_func = lambda p: p-p_thres,
                    value_constraint = lambda x: abs(x)<20,
                    raw_filename = 'result_table_centroid/updeANMO_shift5_catalog_*1[01234]_plot.pkl',
                    summary_ray=True,
                    demean=True,
                    plot=False,
                    include_id=True
                )
get_s_table = lambda gcarc_range: plot_depthslice(phase = 's',
                    value = 'anomaly',
                    gcarc_range = gcarc_range,
                    fidelity_func = lambda p: p-s_thres,
                    value_constraint = lambda x: abs(x)<30,
                    raw_filename = 'result_table_centroid/updeANMO_shift5_catalog_*1[01234]_plot.pkl',
                    summary_ray=True,
                    demean=True,
                    plot=False,
                    include_id=True
                )

p_table = get_p_table(gcarc_range)
s_table = get_s_table(gcarc_range)
p_table = p_table[abs(p_table['arrival_time']-UTCDateTime(2014,10,9,2,14,0))>60*15]
s_table = s_table[abs(s_table['arrival_time']-UTCDateTime(2014,10,9,2,14,0))>60*15]
# df = pd.concat([pd.read_pickle(filename) for filename in glob.glob(f'result_table_centroid/updeANMO_shift5_catalog_*_plot.pkl')], ignore_index=True)
table = plot_ratio(p_table, s_table, gcarc_range, x_dep_y=True, x_label=False, y_label=False, suffix1='_p', suffix2='_s')

# plt.figure()
nrow = 1
mark_size=lambda r: 50*r**2
colorscale = None
plot_colorbar = True
demean = False
bm = []
sc = []
gcarc = []
annot = []

# プロット点をマウスオーバーでラベル表示
fig, axes = plt.subplots(nrow,2, figsize=(24,8))
# fig.figure()
for i in range(2): 
    phase = 's' if i%2 else 'p'
    plot_legend = False if i%2 else True
    lons = table['turning_lon'].values[::nskip]
    lats = table['turning_lat'].values[::nskip]
    prob = table[f'probability_{phase}'].values[::nskip]
    anomaly = table[f'anomaly_{phase}'].values[::nskip]
    avail_pts = sum(table[f'anomaly_{phase}'].notnull())
    axes[i].set_title(f'{phase.upper()} travel time residuals (sec) in {gcarc_range[0]}~{gcarc_range[1]} deg\n{"demeaned" if demean else""} $\delta t_{phase.upper()}$, {avail_pts} pts {f"plotted every {nskip}" if nskip>1 else ""}', fontsize=20, zorder=0)

    
    bm.append(Basemap(projection='moll',lon_0=-180,resolution='c', ax=axes[i]))
    x, y = bm[i](lons, lats)
    # anomaly -= np.nanmean(anomaly)

    bm[i].drawmapboundary(fill_color='white')
    bm[i].drawcoastlines()
    sc.append(bm[i].scatter(x, y, c=anomaly, s=mark_size(prob), marker='o', cmap=cmap))
    # plt.title(f'{phase.upper()}-wave arrival time (sec) in {gcarc_range[0]}~{gcarc_range[1]} deg\n {value}, {sum(scatter_table[value].notnull())} pts')
    
    colorscale = 15 if phase.lower() == 's' else 8
    sc[i].set_clim(-colorscale, colorscale)

    if plot_colorbar:
        cbar = plt.colorbar(sc[i], ax=axes[i], fraction=0.1, aspect=50, pad=0.02, location='bottom')
        cbar.ax.tick_params(labelsize=16)

    if plot_legend:
        sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        markers = [plt.scatter([], [], s=mark_size(size), edgecolors='none', color='dimgray') for size in sizes]
        labels = [f'{size:.1f}' for size in sizes]
        axes[i].legend(markers, labels, title="probability", scatterpoints=1, frameon=False, loc='lower left', bbox_to_anchor=(0.99, 0.75), fontsize=12)


    # gcx, gcy = bm[i].gcpoints()
    gcarc.append(bm[i].drawgreatcircle(0, 0, 10, 10, 50, linewidth=2, color='grey')+
                 bm[i].drawgreatcircle(0, 0, 10, 10, 50, linewidth=2, color='grey'))
    # print(gcarc[i][0])
    for curve in gcarc[i]: curve.set_visible(False)

    annot.append(axes[i].annotate("", xy=(0,0), xytext=(-70 if i%2 else -200,30),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"), zorder=101))
    annot[i].set_visible(False)

def update_annot(sc, bm, annot, gcarc, ind):
    i = ind["ind"][0]
    pos = sc.get_offsets()[i]
    annot.xy = pos
    element = table[::nskip].iloc[i]
    text = str(element).replace('quakeml:jun.su/globocat', '~').replace(' '*10,' '*7)
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(int(text)/10))

    # new_lat = table[::nskip].iloc[i]['turning_lat']
    # new_lon = table[::nskip].iloc[i]['turning_lon']
    # print(gcarc.get_data())
    new_x, new_y = bm.gcpoints(element['origin_lon'], element['origin_lat'], element['station_lon'], element['station_lat'], 100)
    new_x = np.array(new_x); new_y = np.array(new_y)
    for j in range(len(gcarc)):
        # elif (np.diff(new_x)>0).any() and (np.diff(new_x)<0).any():
        #     new_start_pt = np.argmax(np.abs(np.diff(new_x)))+1
        # print(np.abs(np.diff(new_x)))
        if (len(new_x) > 1) and (np.max(np.abs(np.diff(new_x)))>5e5):
            new_start_pt = np.argmax(np.abs(np.diff(new_x)))+1
        elif (new_x[0] > new_x[-1]) and (np.argmax(new_x) != 0):
            new_start_pt = np.argmin(new_x)
        elif (new_x[0] < new_x[-1]) and (np.argmin(new_x) != 0):
            new_start_pt = np.argmax(new_x)
        else:
            new_start_pt = None

        # print(np.abs(np.diff(new_x))[:new_start_pt])
        # print(new_x[:new_start_pt])
        gcarc[j].set_data(new_x[:new_start_pt], new_y[:new_start_pt])
        new_x = new_x[new_start_pt:]
        new_y = new_y[new_start_pt:]
    # print('----')
    # print(gcarc.get_data())
    # print(new_x)
    # gcarc[-1].set_data(new_x, new_y)

def hover(event):
    # print(event.inaxes == axes[0], event.inaxes == axes[1])
    event_inaxes = None
    for i in range(2):
        if event.inaxes == axes[i]: event_inaxes = i
    if event_inaxes is not None:
        # print(event_inaxes)
        vis = annot[event_inaxes].get_visible()
        cont, ind = sc[event_inaxes].contains(event)
        if cont:
            update_annot(sc[event_inaxes], bm[event_inaxes], annot[event_inaxes], gcarc[event_inaxes], ind)
            for j in range(len(annot)):
                    annot[j].set_visible(True if j==event_inaxes else False)
                    for curve in gcarc[j]: curve.set_visible(True if j==event_inaxes else False)
            # annot[event_inaxes].set_visible(True)
            fig.canvas.draw_idle()
            # print(event.x)
        else:
            if vis:
                for j in range(len(annot)):
                    annot[j].set_visible(False)
                    for curve in gcarc[j]: curve.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
fig.subplots_adjust(wspace=0.1)
plt.show()