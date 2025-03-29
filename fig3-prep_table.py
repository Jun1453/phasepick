from export_catalog import plot_depthslice, plot_ratio
p_thres = 0.7
s_thres = 0.5
                
gcarc_range = (0, 180)

p_table = plot_depthslice(phase = 'p',
            value = 'anomaly_rev',
            gcarc_range = gcarc_range,
            fidelity_func = lambda p: p-p_thres,
            value_constraint = lambda x: abs(x)<20,
            raw_filename = './result_table_centroid/updeANMO_shift5_catalog_*_plot.pkl',
            summary_ray=True,
            demean=True,
            plot=False,
            include_id = True
        )
s_table = plot_depthslice(phase = 's',
    value = 'anomaly_rev',
    gcarc_range = gcarc_range,
    fidelity_func = lambda p: p-s_thres,
    value_constraint = lambda x: abs(x)<30,
    raw_filename = './result_table_centroid/updeANMO_shift5_catalog_*_plot.pkl',
    summary_ray=True,
    demean=True,
    plot=False,
    include_id = True
)

p_table.to_pickle('su2024grl-scatter/p_table_isc-origin.pkl')
s_table.to_pickle('su2024grl-scatter/s_table_isc-origin.pkl')