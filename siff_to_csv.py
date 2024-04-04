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
    if ~siff_path.exists():
        raise FileNotFoundError(f"{siff_path} does not exist")
    
    if ~siff_path.suffix == ".siff":
        raise ValueError("The file must be a `.siff` file")
    
    sr = SiffReader(siff_path)
    tzero = sr.tzero
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
    )->Tuple[np.ndarray, np.ndarray]:
    """
    """
    fluorescence = sr.get_frames(frames = sr.im_params.flatten_by_timepoints()).astype(float).reshape(sr.im_params.array_shape).squeeze()

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

    return fluorescence, flim_trace_blurred

def siff_to_series(
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

    tzero = sr.tzero

    t_axis = sr.t_axis(reference_time='epoch')

    fluorescence, flim_trace_blurred = to_traces(sr, rois)

def plot_timeseries(
    sr : SiffReader,
    exp : Experiment,
    t_axis : np.ndarray,
    fluorescence : np.ndarray,
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
    save_path : Path
):
    """
    """

    import pandas as pd

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4,1.5))

    df_together = pd.read_excel(Path(f'/Users/stephen/Documents/Manuscripts/Demotivation/Nat 2023 Review/Feb24/dopamine_imaging_data/')/ 'summary_trp.xlsx')

    x_jit = np.random.normal(0, 0.03, len(df_together))
    ax.plot(x_jit, df_together.flim_20, 'o', color = BRIGHT_VIOLET, alpha = 0.5, markersize = 4)

    for x in range(len(df_together)):
        ax.plot([x_jit[x], 1+x_jit[x]], [df_together.flim_20[x], df_together.flim_30[x]], color = 'k', alpha = 0.5)

    ax.plot(1+x_jit, df_together.flim_30, 'o', color = BRIGHT_VIOLET, alpha = 1.0, markersize = 4)

    ax.plot([-0.15, 0.15], np.mean(df_together.flim_20)*np.ones(2), color = 'k', alpha = 1.0, linewidth = 3)
    ax.plot(
        [0,0],
        [
            np.mean(df_together.flim_20)-np.std(df_together.flim_20)/np.sqrt(len(df_together.flim_20)),
            np.mean(df_together.flim_20)+np.std(df_together.flim_20)/np.sqrt(len(df_together.flim_20))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )
    ax.plot([0.85, 1.15], np.mean(df_together.flim_30)*np.ones(2), color = 'k', alpha = 1.0, linewidth = 3)
    ax.plot(
        [1,1],
        [
            np.mean(df_together.flim_30)-np.std(df_together.flim_30)/np.sqrt(len(df_together.flim_30)),
            np.mean(df_together.flim_30)+np.std(df_together.flim_30)/np.sqrt(len(df_together.flim_30))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )

    df_together = pd.read_excel(Path(f'/Users/stephen/Documents/Manuscripts/Demotivation/Nat 2023 Review/Feb24/dopamine_imaging_data/')/ 'summary_wt.xlsx')

    x_jit = 2+np.random.normal(0, 0.03, len(df_together))
    ax.plot(x_jit, df_together.flim_20, 'o', color = '#000000', alpha = 0.5, markersize = 4)

    for x in range(len(df_together)):
        ax.plot([x_jit[x], 1+x_jit[x]], [df_together.flim_20[x], df_together.flim_30[x]], color = 'k', alpha = 0.5)

    ax.plot(1+x_jit, df_together.flim_30, 'o', color = '#000000', alpha = 1.0, markersize = 4)

    ax.plot([1.85, 2.15], np.mean(df_together.flim_20)*np.ones(2), color = 'k', alpha = 1.0, linewidth = 3)
    ax.plot(
        [2,2],
        [
            np.mean(df_together.flim_20)-np.std(df_together.flim_20)/np.sqrt(len(df_together.flim_20)),
            np.mean(df_together.flim_20)+np.std(df_together.flim_20)/np.sqrt(len(df_together.flim_20))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )

    ax.plot([2.85, 3.15], np.mean(df_together.flim_30)*np.ones(2), color = 'k', alpha = 1.0, linewidth = 3)

    ax.plot(
        [3,3],
        [
            np.mean(df_together.flim_30)-np.std(df_together.flim_30)/np.sqrt(len(df_together.flim_30)),
            np.mean(df_together.flim_30)+np.std(df_together.flim_30)/np.sqrt(len(df_together.flim_30))
        ], color = 'k', alpha = 1.0, linewidth = 3
    )


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    #ax.set_yticks([-0.5, 0, 0.5, 1.0])

    fig.savefig(
        Path(save_path)/'summary.pdf',
    )