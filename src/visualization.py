import copy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase, make_axes

from trajectory_analysis_tools import get_trajectory_data, get_distance_metrics


def make_movie(time_slice, classifier, results, data, frame_rate=500,
               movie_name="video_name.mp4", position_color="magenta",
               map_color="limegreen"
              ):
    t = data["position_info"].index / np.timedelta64(1, "s")
    posterior = (results["acausal_posterior"]
                 .sum("state", skipna=False)
                 .sel(time=time_slice))
    (actual_projected_position, actual_edges, directions,
     map_position, map_edges) = get_trajectory_data(
        posterior=posterior,
        track_graph=data["track_graph"],
        decoder=classifier,
        position_info=data["position_info"].reset_index().set_index(t).loc[time_slice]
    )
    
    
    # Set up formatting for the movie files
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=frame_rate, metadata=dict(artist="Me"), bitrate=1800)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_facecolor('black')
    ax.plot(
        data["position_info"].x_position,
        data["position_info"].y_position,
        color="lightgrey",
        alpha=0.4,
        zorder=1,
    )

    ax.set_xlim(data["position_info"].x_position.min() - 1,
                data["position_info"].x_position.max() + 1)
    ax.set_ylim(data["position_info"].y_position.min() + 1,
                data["position_info"].y_position.max() + 1)
    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")

    position_dot = plt.scatter(
        [], [], s=80, zorder=101, color=position_color, label="Actual"
    )
    position_arrow = ax.arrow(
        [], [], [], [],
        zorder=102,
        head_width=1.5,
        linewidth=3,
        color=position_color)
    (position_line,) = plt.plot([], [], color=position_color, linewidth=3)
    sns.despine()
    map_dot = plt.scatter([], [], s=80, zorder=102, color=map_color, label="Decoded")
    (map_line,) = plt.plot([], [], color=map_color, linewidth=3)
    ax.legend(fontsize=9, loc='upper right')
    n_frames = map_position.shape[0]

    def _update_plot(time_ind):
        start_ind = max(0, time_ind - 5)
        time_slice = slice(start_ind, time_ind)

        ax.patches.pop(0)
        patch = ax.arrow(actual_projected_position[time_ind, 0], 
                         actual_projected_position[time_ind, 1],
                         1e-5 * np.cos(directions[time_ind]),
                         1e-5 * np.sin(directions[time_ind]),
                         zorder=102,
                         head_width=1.5,
                         linewidth=3,
                         color=position_color, 
                         )
        position_dot.set_offsets(actual_projected_position[time_ind])
        position_line.set_data(
            actual_projected_position[time_slice, 0],
            actual_projected_position[time_slice, 1])

        map_dot.set_offsets(map_position[time_ind])
        map_line.set_data(
            map_position[time_slice, 0],
            map_position[time_slice, 1])

        return position_arrow, map_dot

    movie = animation.FuncAnimation(
        fig, _update_plot, frames=n_frames, interval=1000 / frame_rate,
        blit=True
    )
    if movie_name is not None:
        movie.save(movie_name, writer=writer)

    return fig, movie


def plot_classifier_time_slice(
    time_slice,
    classifier,
    results,
    data,
    posterior_type="acausal_posterior",
    figsize=(30, 20),
    cmap="bone_r",
):

    t = data["position_info"].index / np.timedelta64(1, "s")
    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        nrows=6,
        ncols=1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1, 1]},
    )

    # ax 0
    posterior = (results[posterior_type]
                 .sum("state", skipna=False)
                 .sel(time=time_slice))

    (posterior.plot(
        x="time", y="position", robust=True, ax=axes[0],
        cmap=cmap, vmin=0.0,
    ))

    axes[0].set_ylabel("Position [cm]")

    axes[0].set_title("Posterior")

    axes[0].plot(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="magenta",
        linestyle="--",
        linewidth=5,
        alpha=0.8,
    )
    axes[0].set_xlabel("")

    # ax 1
    results[posterior_type].sum("position").sel(time=time_slice).plot(
        x="time", hue="state", ax=axes[1],
    )
    axes[1].set_title("Probability")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("")

    # ax 2
    trajectory_data = get_trajectory_data(
        posterior=posterior,
        track_graph=data["track_graph"],
        decoder=classifier,
        position_info=data["position_info"].reset_index().set_index(t).loc[time_slice]
    )

    distance_metrics = get_distance_metrics(data["track_graph"], *trajectory_data)
    ahead_behind_distance = (
        distance_metrics.mental_position_ahead_behind_animal *
        distance_metrics.mental_position_distance_from_animal)
    axes[2].plot(posterior.time, ahead_behind_distance, color="black", linewidth=2)
    axes[2].axhline(0, color="magenta", linestyle="--")
    axes[2].set_title("Mental distance ahead or behind animal")
    axes[2].set_ylabel("Distance [cm]")
    max_dist = np.max(distance_metrics.mental_position_distance_from_animal) + 5
    axes[2].set_ylim((-max_dist, max_dist))
    axes[2].text(posterior.time[0], max_dist-1, "Ahead", color="grey")
    axes[2].text(posterior.time[0], -max_dist+1, "Behind", color="grey")
    # ax 3
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[3].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[3].set_ylabel("Firing Rate\n[spikes / s]")
    axes[3].set_title("Multiunit")

    # ax 4
    theta = data['theta'].set_index(
        data['theta'].index / np.timedelta64(1, 's')).loc[time_slice]
    axes[4].plot(theta.index, theta.bandpassed_lfp)
    axes[4].set_title('Theta filtered LFP')
    axes[4].set_ylabel('Amplitude [mV]')

    # ax 5
    axes[5].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[5].set_title('Speed')
    axes[5].set_ylabel("Speed [cm / s]")
    axes[5].set_xlabel("Time [s]")
    sns.despine()


def plot_local_non_local_time_slice(
    time_slice,
    detector,
    results,
    data,
    posterior_type="acausal_posterior",
    cmap="bone_r",
    figsize=(30, 20),
):
    t = data["position_info"].index / np.timedelta64(1, "s")
    mask = np.ones_like(detector.is_track_interior_.squeeze(), dtype=np.float)
    mask[~detector.is_track_interior_] = np.nan
    cmap = copy.copy(plt.cm.get_cmap(cmap))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1, 1]},
    )

    # ax 0
    (results[posterior_type].sel(time=time_slice).sum("state") * mask).plot(
        x="time", y="position", robust=True, ax=axes[0], cmap=cmap, vmin=0.0,
    )
    axes[0].set_ylabel("Position [cm]")

    axes[0].set_title("Non-Local Posterior")

    axes[0].plot(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="white",
        linestyle="--",
        linewidth=5,
        alpha=0.8,
    )
    axes[0].set_xlabel("")

    # ax 1
    results[posterior_type].sum("position").sel(
        state="Non-Local", time=time_slice
    ).plot(x="time", ax=axes[1])
    axes[1].set_title("Non-Local Probability")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("")

    # ax 2
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[2].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[2].set_ylabel("Firing Rate\n[spikes / s]")
    axes[2].set_title("Multiunit")

    # ax 4
    theta = data['theta'].set_index(
        data['theta'].index / np.timedelta64(1, 's')).loc[time_slice]
    axes[3].plot(theta.index, theta.bandpassed_lfp)
    axes[3].set_title('Theta filtered LFP')
    axes[3].set_ylabel('Amplitude [mV]')

    # ax 5
    axes[4].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .speed.values.squeeze(),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[4].set_ylabel("Speed [cm / s]")
    axes[4].set_title('Speed')
    axes[4].set_xlabel("Time [s]")
    sns.despine()


def plot_2D_position_with_color_time(time, position, ax=None, cmap='plasma',
                                     alpha=None):
    '''

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    position : ndarray, shape (n_time, 2)
    ax : None or `matplotlib.axes.Axes` instance
    cmap : str
    alpha : None or ndarray, shape (n_time,)

    Returns
    -------
    line : `matplotlib.collections.LineCollection` instance
    ax : `matplotlib.axes.Axes` instance

    '''
    if ax is None:
        ax = plt.gca()
    points = position.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(vmin=time.min(), vmax=time.max())
    cmap = plt.get_cmap(cmap)
    colors = cmap(norm(time))
    if alpha is not None:
        colors[:, -1] = alpha

    lc = LineCollection(segments, colors=colors, zorder=100)
    lc.set_linewidth(6)
    line = ax.add_collection(lc)

    # Set the values used for colormapping
    cax, _ = make_axes(ax, location='bottom')
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm,
                        spacing='proportional',
                        orientation='horizontal')
    cbar.set_label('Time')

    total_distance_traveled = np.linalg.norm(
        np.diff(position, axis=0), axis=1).sum()
    if np.isclose(total_distance_traveled, 0.0):
        ax.scatter(position[:, 0], position[:, 1],
                   c=colors, zorder=1000, s=70, marker='s')

    return line, ax, cbar
