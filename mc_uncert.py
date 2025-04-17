import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor


def get_ratio(scatter_table, demean=True, x_dep_y=True, slope_two_sections=False):
    x = scatter_table[f'anomaly{1 if not x_dep_y else 2}'].values
    y = scatter_table[f'anomaly{2 if not x_dep_y else 1}'].values
    if demean:
        x -= np.mean(x)
        y -= np.mean(y)

    if slope_two_sections:
        slope = [None, None]
        x_bound = [x<=0, x>=0]
    else:
        slope = [None]
        x_bound = [x != np.nan]

    for i in range(len(slope)):
        fit_range = np.array([min(x[x_bound[i]]), max(x[x_bound[i]])])
        theilsen = TheilSenRegressor(random_state=42).fit(x[x_bound[i]].reshape(-1,1), y[x_bound[i]])
        slope[i] = round(np.diff(theilsen.predict(fit_range.reshape(-1,1)))[0] / np.diff(fit_range)[0], 4)

    return np.round(np.reciprocal(slope), 4) if x_dep_y else slope

z = np.arange(30,100,5)
sample_num = np.ones(len(z))
ratio = np.ones(len(z))
ratio_5p = np.ones(len(z))
ratio_95p = np.ones(len(z))
for i in range(len(z)):
    tb = pd.read_pickle(f'/Users/junsu/phasepick/su2024grl-scatter/{z[i]}-{z[i]+5}.pkl')
    sample_num[i] = len(tb)
    #ratio[i] = get_ratio(tb)[0]
    #ratio_smp = np.array([get_ratio(tb.sample(int(0.05*len(tb))))[0] for _ in range(200)])
    #ratio_5p[i] = np.percentile(ratio_smp, 5)
    #ratio_95p[i] = np.percentile(ratio_smp, 95)

np.save('sample_num.py', sample_num)
#np.save('ratio.npy', ratio)
#np.save('ratio_5p.npy', ratio_5p)
#np.save('ratio_95p.npy', ratio_95p)
