# %%
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm
import numpy as np

# %%
# I/O
inflow = np.loadtxt('data/djk_100ensemble.txt', dtype=float, skiprows=1) # inflow ensemble forecast
f1 = np.loadtxt('data/perf_f1.csv', dtype=float, skiprows=0, delimiter=",") # performance values of candidate solutions from simulation
# inflow: (number of months/timesteps, number of ensemble members + 3 titles columns)
# performance: (number of solutions, number of ensemble members)

# %%
# data initialisation
selected_sol = range(f1.shape[0]) # number of candidate solutions
num_data = inflow.shape[1] - 3  # number of ensemble members
ep_0 = np.arange(1, num_data + 1) / num_data # exceedance probability array init

# %%
# functions
def find_pth(x, y, pth):
    target = np.percentile(y, pth)
    ix = np.abs(y - target).argmin()
    return x[ix], y[ix]

def find_mean(x, y):
    target = np.mean(y)
    ix = np.abs(y - target).argmin()
    return x[ix], y[ix]

def find_statistics(long_fx, statistics, pth):
    list_v_fx = []
    list_i_fx = []
    for m in selected_sol:
        if statistics == "mean":
            ind_fx, val_fx = find_mean(ep_0, long_fx[m])
        else:
            ind_fx, val_fx = find_pth(ep_0, long_fx[m], pth)
        list_v_fx.append(val_fx)
        list_i_fx.append(ind_fx)
    return list_v_fx, list_i_fx

def get_rankings(array):
    return np.argsort(array) + 1

def filter_posit(data):
    return data[data > 0]

def calc_risk(long_fx):
    list_risk = []
    for m in selected_sol:
        data = long_fx[m,:]
        posit_data = filter_posit(data) # only positive deficits
        num_posit_data = len(posit_data)
        sorted_indices = np.argsort(posit_data)[::-1]
        sorted_data = posit_data[sorted_indices]
        ep = np.arange(1, num_posit_data + 1) / num_data
        area = np.trapz(sorted_data, ep)
        list_risk.append(area)
    return list_risk

# %%
# plotting aesthetics
# matplotlib parameters
plt.rcParams['figure.dpi'] = 600 # dpi in ipy
plt.rcParams["font.family"] = "Helvetica Neue"
plt.rcParams.update({'font.size': 11})
cm_trans = 1/2.54 # cm to inch

# color spectrum
cs = cm.coolwarm(np.linspace(0, 1, 5))
cs[2] = [0.6,0.6,0.6,0.5].copy() # grey color

# alpha and scatter size
val_alpha1 = 1 # solid line
val_alpha2 = 0.9 # scatter projections
val_alpha3 = 0.3 # region
size_sca = 60

# %%
# trade-off map plotting
fig, ax1 = plt.subplots(1, 1, figsize=(11.23*1.4*cm_trans, 8.31*cm_trans), dpi=300)
ax1.ticklabel_format(axis="y", useMathText=True)
ax1.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
x_locations = np.array([0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
y_locations = np.array([0.0, 200, 400, 600, 800])
ax1.set_xticks(x_locations)
ax1.set_yticks(y_locations)
ax1.set_xlim([0,1])
ax1.set_ylim([0,800])

# trade-off curve plotting
for m in selected_sol:
    posit_fxm = filter_posit(f1[m,:]) # only positive deficits
    y = posit_fxm
    x = ep_0[:len(y)] # corresponding exceedance probabilities
    ax1.plot(x, y, color=cs[m], alpha=val_alpha1, zorder=1)

# statistics marking
# estimate values and positions of statistics on trade-off curve
list_v_fx_wc,   list_i_fx_wc   = find_statistics(f1, "wc",   100)
list_v_fx_wp10, list_i_fx_wp10 = find_statistics(f1, "wp10", 91)
list_v_fx_wp25, list_i_fx_wp25 = find_statistics(f1, "wp25", 76)
list_v_fx_mean, list_i_fx_mean = find_statistics(f1, "mean", 100)
list_v_fx_bc,   list_i_fx_bc   = find_statistics(f1, "bc",   0)

# plot scatters of statistics
for m in selected_sol:
    ax1.scatter(list_i_fx_wc[m],   list_v_fx_wc[m],   marker="p", s=size_sca, facecolors=cs[m], edgecolors="none", alpha=val_alpha2) # WC
    ax1.scatter(list_i_fx_wp10[m], list_v_fx_wp10[m], marker="d", s=size_sca, facecolors=cs[m], edgecolors="none", alpha=val_alpha2) # WP10
    ax1.scatter(list_i_fx_wp25[m], list_v_fx_wp25[m], marker="^", s=size_sca, facecolors=cs[m], edgecolors="none", alpha=val_alpha2) # WP25
    ax1.scatter(list_i_fx_mean[m], list_v_fx_mean[m], marker="x", s=size_sca*0.7, color=cs[m], alpha=val_alpha2) # mean
    ax1.scatter(list_i_fx_bc[m],   list_v_fx_bc[m],   marker="o", s=size_sca, facecolors=cs[m], edgecolors="none", alpha=val_alpha2) # BC

plt.tight_layout()
plt.savefig('fig_tradeoff.jpg', dpi=600, transparent=True)

# %%
# rankings based on different statistics
rank_fx_wc = get_rankings(list_v_fx_wc)
rank_fx_wp10 = get_rankings(list_v_fx_wp10)
rank_fx_wp25 = get_rankings(list_v_fx_wp25)
rank_fx_mean = get_rankings(list_v_fx_mean)

print("Rankings of candidate solutions based on different statistics:")
print(rank_fx_wc)
print(rank_fx_wp10)
print(rank_fx_wp25)
print(rank_fx_mean)

# %%
# risk value calculation
# trapezoidal approximation of the area under the trade-off curve
list_fx_risk = calc_risk(f1)
rank_fx_risk = get_rankings(list_fx_risk)

print("Risk values of candidate solutions:")
print(list_fx_risk)
print("Rankings of candidate solutions:")
print(rank_fx_risk)
# %%