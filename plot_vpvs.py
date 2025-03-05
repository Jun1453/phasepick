import matplotlib.pyplot as plt
from export_catalog import plot_depthslice, plot_vpvs_ratio
p_thres = 0.7
s_thres = 0.5

get_p_table = lambda gcarc_range: plot_depthslice(phase = 'p',
                    value = 'anomaly',
                    gcarc_range = gcarc_range,
                    fidelity_func = lambda p: p-p_thres,
                    value_constraint = lambda x: abs(x)<20,
                    raw_filename = 'updeANMO_shift5_catalog_*_plot.pkl',
                    average_ray=True,
                    demean=True,
                    plot=False
                )
get_s_table = lambda gcarc_range: plot_depthslice(phase = 's',
                    value = 'anomaly',
                    gcarc_range = gcarc_range,
                    fidelity_func = lambda p: p-s_thres,
                    value_constraint = lambda x: abs(x)<30,
                    raw_filename = 'updeANMO_shift5_catalog_*_plot.pkl',
                    average_ray=True,
                    demean=True,
                    plot=False
                )


plt.figure(figsize=(12,26))
plt.subplot(421); plot_vpvs_ratio(get_p_table, get_s_table, (50, 55), plot_ylabel=True)
plt.subplot(423); plot_vpvs_ratio(get_p_table, get_s_table, (55, 60), plot_ylabel=True)
plt.subplot(425); plot_vpvs_ratio(get_p_table, get_s_table, (60, 65), plot_ylabel=True)
plt.subplot(427); plot_vpvs_ratio(get_p_table, get_s_table, (65, 70), plot_ylabel=True, plot_xlabel=True)
plt.subplot(422); plot_vpvs_ratio(get_p_table, get_s_table, (70, 75))
plt.subplot(424); plot_vpvs_ratio(get_p_table, get_s_table, (75, 80))
plt.subplot(426); plot_vpvs_ratio(get_p_table, get_s_table, (80, 85))
plt.subplot(428); plot_vpvs_ratio(get_p_table, get_s_table, (85, 90), plot_xlabel=True)
plt.subplots_adjust(hspace=0.2)
plt.savefig("qc_vpvs.50-90.png", bbox_inches="tight")