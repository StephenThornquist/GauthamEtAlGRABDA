"""
A generous, but disingenuous, retelling of the path from data
to figure. This script contains methods that analyze and format
a set of `.siff` data files and package summary statistics into
a single `.csv` file that is read in the `basic_analysis` notebook.
"""

from typing import Optional, Tuple, List
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


from siffpy import SiffReader
from sifftrac.ros import find_experiment_containing_timestamp, Experiment
from siffroi import load_rois, ROI

from siffplot.sifftrac.events import add_event_axis_matplotlib
from siffplot.siffpy import plot_siff_events
from siffplot.sifftrac import plot_temperature

BRIGHT_VIOLET = '#9467bd'

def load_data(
    siff_path : Path,
    )->Tuple[SiffReader, Experiment, List[ROI]]:
    """

    """
    # I/O boilerplate
    siff_path = Path(siff_path)
    
    if siff_path.suffix != ".siff":
        raise ValueError("The file must be a `.siff` file")
    
    sr = SiffReader(siff_path)
    tzero = sr.time_zero
    # Find the temperature control data
    exp = find_experiment_containing_timestamp(
        sr.filename.parent.parent.parent,
        tzero
    )

    rois = load_rois(sr.filename.parent)

    return sr, exp, rois

def to_traces(
    sr : SiffReader,
    rois : List[ROI],
    )->np.ndarray:
    """
    """
    flim_trace = sr.sum_mask_flim(
        sr.flim_params[0],
        rois[0].mask if len(rois) > 0 else np.ones(sr.im_params.volume).squeeze().astype(bool),
    )
    
    flim_trace.convert_units('nanoseconds')

    flim_trace_blurred = np.convolve(
        flim_trace,
        np.ones(int(1.0/sr.dt_frame)).astype(float)/(int(1.0/sr.dt_frame)),
        mode='same'
    ) 

    return flim_trace_blurred

def siff_to_df(
    siff_path : Path,
    save_path : Path,
    )->SiffReader:
    """
    Saves the data from a single `.siff` file into the
    specified `.csv` file in the `save_path`

    Parameters
    ----------

    siff_path : Path
        The path to the `.siff` file to be read

    save_path : Path
        The path to the `.csv` file to be saved

    Returns
    -------

    sr : SiffReader
        The `SiffReader` object that was used to read the data
        if you want to plot the data too
    """
    sr, exp, rois = load_data(siff_path)
    
    if ~save_path.suffix == ".csv":
        save_path = save_path.with_suffix(".csv")

    if ~save_path.parent.exists():
        save_path.mkdir(parents=True)

    t_axis = sr.t_axis(reference_time='epoch')

    flim_trace_blurred = to_traces(sr, rois)


    fo = flim_trace_blurred.intensity[int(5/sr.dt_frame):int(35/sr.dt_frame)].mean()
    f = flim_trace_blurred.intensity

    dfof = (f-fo)/fo


    crop_temp = exp.warner_temperature.temperature[(exp.warner_temperature.timestamps > sr.time_zero) * (exp.warner_temperature.timestamps < t_axis[-1])].values
    crop_t = exp.warner_temperature.timestamps[(exp.warner_temperature.timestamps > sr.time_zero) * (exp.warner_temperature.timestamps < t_axis[-1])].values
    dt_temp = (exp.warner_temperature.timestamps[1] - exp.warner_temperature.timestamps[0])/1e9

    temp_increasing = np.convolve(np.diff(crop_temp), np.ones(int(15/dt_temp)).astype(float)/int(15/dt_temp), mode='same')

    temp_twodir = np.insert(np.convolve(np.diff(temp_increasing), np.ones(int(15/dt_temp)).astype(float)/int(15/dt_temp), mode='same'), 0,0)

    increasing_start = np.where(
        (temp_increasing > (np.mean(temp_increasing) + np.std(temp_increasing))/2)
        & (temp_twodir > (np.mean(temp_twodir) + np.std(temp_twodir)))
    )[0][0]

    increasing_end = np.where(
        (temp_increasing > (np.mean(temp_increasing) + np.std(temp_increasing))/2)
        & (temp_twodir < (np.mean(temp_twodir) - np.std(temp_twodir))) &
        (np.arange(len(temp_twodir)) > increasing_start)
    )[0][0]

    decreasing_start = np.where(
        (temp_increasing < (np.mean(temp_increasing) - np.std(temp_increasing))/2)
        & (temp_twodir < (np.mean(temp_twodir) - np.std(temp_twodir))/2)
        & (np.arange(len(temp_twodir)) > increasing_end)
    )[0][0]

    decreasing_end = np.where(
        (temp_increasing < (np.mean(temp_increasing) - np.std(temp_increasing))/2)
        & (temp_twodir > (np.mean(temp_twodir) + np.std(temp_twodir))/2)
        & (np.arange(len(temp_twodir)) > decreasing_start)
    )[0][0]

    inc_start_t, inc_end_t, dec_start_t, _ = crop_t[increasing_start], crop_t[increasing_end], crop_t[decreasing_start], crop_t[decreasing_end]

    data= [
        np.mean(
            dfof[np.where((t_axis < inc_start_t) & (t_axis > (inc_start_t - 30*1e9)))[0]]
        ),
        np.mean(
            dfof[np.where((t_axis < dec_start_t) & (t_axis > inc_end_t))[0]]
        ),

        np.mean(
            flim_trace_blurred[np.where((t_axis < inc_start_t) & (t_axis > (inc_start_t - 30*1e9)))[0]]
        ).lifetime,

        np.mean(
            flim_trace_blurred[np.where((t_axis < dec_start_t) & (t_axis > inc_end_t))[0]]
        ).lifetime
    ]

    if ~(save_path.parent.exists()):
        save_path.parent.mkdir(parents=True)
        pd.DataFrame(
            data = [data],
            columns = ['flim_20', 'flim_30', 'flim_20_lifetime', 'flim_30_lifetime']
        ).to_csv(save_path, index = False)
    else:
        df = pd.read_csv(save_path)
        # append the new data to the end of the df
        df.loc[len(df)] = data
        df.to_csv(save_path, index = False)

def plot_timeseries(
    sr : SiffReader,
    exp : Experiment,
    t_axis : np.ndarray,
    flim_trace_blurred : np.ndarray,
    save_path : Optional[Path] = None,
)->Tuple[Figure, Axes]:
    """
    If `save_path` is None, does not save the plot
    """
    from matplotlib import rcParams
    rcParams['font.size'] = 9
    rcParams['font.family'] = 'Arial'
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False

    [stamp.timestamp_epoch for stamp in sr.get_appended_text()]

    tmp_f, tmp_x = plt.subplots(nrows = 3, sharex = True, figsize = (3,0.5))

    tmp_f.set_size_inches(4,1)

    tmp_x[0].plot((t_axis - sr.time_zero)/1e9, flim_trace_blurred.lifetime, color = BRIGHT_VIOLET)
    tmp_x[0].set_ylabel('Lifetime (ns)')

    fo = flim_trace_blurred.intensity[int(5/sr.dt_frame):int(35/sr.dt_frame)].mean()
    f = flim_trace_blurred.intensity

    dfof = (f-fo)/fo

    tmp_x[1].plot((t_axis - sr.time_zero)/1e9, dfof, color = BRIGHT_VIOLET)
    tmp_x[1].set_ylabel(r'$\Delta F/F_0$')

    tmp_x[1].plot(
        tmp_x[1].get_xlim(),
        [0,0],
        color = 'k',
        linestyle = '--',
        alpha = 0.3,
        zorder = -1
    )

    event_x = add_event_axis_matplotlib(tmp_f, tmp_x[0], location = 'top')

    imaging_xlim = tmp_x[0].get_xlim()

    plot_siff_events(sr, event_x, sr.time_zero)
    event_x.set_xlim(0, imaging_xlim[1])

    plot_temperature(exp.warner_temperature, tmp_x[2], sr.time_zero, )
    tmp_x[2].set_xlim(*imaging_xlim)
    tmp_x[2].set_ylim(15, 32)

    tmp_x[0].set_ylim(1.6, 2.3)
    tmp_x[1].set_ylim(-0.5,1.0)


    tmp_f.set_size_inches(10,6)

    tmp_x[-1].set_xlim(0, tmp_x[0].get_xlim()[1])
    tmp_x[-1].plot(
        [np.mean(tmp_x[2].get_xlim()), np.mean(tmp_x[2].get_xlim())+60],
        [np.mean(tmp_x[2].get_ylim()), np.mean(tmp_x[2].get_ylim())],
        color = 'k'
    )

    tmp_x[0].set_title(sr.filename.stem)

    for x in tmp_x:
        x.spines['top'].set_visible(False)
        x.spines['right'].set_visible(False)
        x.spines['bottom'].set_visible(False)
        x.set_xticks([])

    tmp_x[1].set_yticks([])
    tmp_x[1].spines['left'].set_visible(False)
    tmp_x[1].plot(
        [np.mean(tmp_x[2].get_xlim()), np.mean(tmp_x[2].get_xlim())],
        [0, 0.5],
        color = 'k'
    )
    tmp_x[1].text(
        np.mean(tmp_x[2].get_xlim())+20,
        0.25,
        '0.5\n$\Delta F/F_0$',
        ha = 'center',
        va = 'center'
    )

    tmp_f.tight_layout()

    if save_path is not None:
        tmp_f.savefig(
            save_path,
            bbox_inches = 'tight'
        )

def plot_summary(
    csv_path : Path,
    color : str = BRIGHT_VIOLET,
    save_path : Optional[Path] = None,
    ax: Optional[Axes] = None,
    condition_1 = 'flim_20',
    condition_2 = 'flim_30',
):
    """
    Plots a .csv file that contains summary statistics
    """
    return_fig = False
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4,1.5))

    df_together = pd.read_excel(csv_path)

    x_jit = np.random.normal(0, 0.03, len(df_together))
    ax.plot(x_jit, df_together[condition_1], 'o', color = color, alpha = 0.5, markersize = 4)

    for x in range(len(df_together)):
        ax.plot(
            [x_jit[x], 1+x_jit[x]],
            [df_together[condition_1][x], df_together[condition_2][x]],
            color = 'k', alpha = 0.5
        )

    ax.plot(1+x_jit, df_together[condition_2], 'o', color = color, alpha = 1.0, markersize = 4)

    ax.plot(
        [-0.15, 0.15],
        np.mean(df_together[condition_1])*np.ones(2),
        color = 'k', alpha = 1.0, linewidth = 3
    )
    ax.plot(
        [0,0],
        [
            np.mean(df_together[condition_1])-np.std(df_together[condition_1])/np.sqrt(len(df_together[condition_1])),
            np.mean(df_together[condition_1])+np.std(df_together[condition_1])/np.sqrt(len(df_together[condition_1]))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )
    ax.plot([0.85, 1.15], np.mean(df_together[condition_2])*np.ones(2), color = 'k', alpha = 1.0, linewidth = 3)
    ax.plot(
        [1,1],
        [
            np.mean(df_together[condition_2])-np.std(df_together[condition_2])/np.sqrt(len(df_together[condition_2])),
            np.mean(df_together[condition_2])+np.std(df_together[condition_2])/np.sqrt(len(df_together[condition_2]))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches = 'tight'
        )

    if return_fig:
        return fig, ax
