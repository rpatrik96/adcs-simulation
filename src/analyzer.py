from os import listdir
from os.path import join, dirname, abspath
from typing import Optional, List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt, rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from kalman_filter import scale_magnet_sun, inv_clip_esoq_q_norm
from logger import Logger
from utils import make_dir


def plot_typography(usetex: bool = False, small: int = 16, medium: int = 20, big: int = 22):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).

    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc('text', usetex=usetex)
    rc('font', family='serif')

    # font sizes
    rc('font', size=small)  # controls default text sizes
    rc('axes', titlesize=big)  # fontsize of the axes title
    rc('axes', labelsize=medium)  # fontsize of the x and y labels
    rc('xtick', labelsize=small)  # fontsize of the tick labels
    rc('ytick', labelsize=small)  # fontsize of the tick labels
    rc('legend', fontsize=small)  # legend fontsize
    rc('figure', titlesize=big)  # fontsize of the figure title


def plot_init(figsize: Tuple[int, int] = (8, 6), inset: bool = True, zoom: float = 2.5, loc: int = 4):
    """
    Initializes the plot including size, grid, margins, and inset (if present)

    :param figsize: tuple of two ints ti set figure size
    :param inset: flag to plot an inset
    :param zoom: zoom factor
    :param loc: location of the inset
    :return:
    """

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # label format, grid, and margins
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, tick_num: int(val*self.decimate_step)))
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=False)
    ax.grid(True, which="both", ls="-.")
    ax.margins(x=0.01, y=0.035)

    # location for the inset
    if loc == 1:
        bbox_to_anchor = (0.95, 0.95)
        loc1, loc2 = 3, 4
    elif loc == 2:
        bbox_to_anchor = (0.08, .95)
        loc1, loc2 = 1, 3
    elif loc == 4:
        bbox_to_anchor = (0.95, .1)
        loc1, loc2 = 2, 4

    # create inset
    if inset:
        axins = zoomed_inset_axes(ax, zoom=zoom, loc=loc, bbox_to_anchor=bbox_to_anchor,
                                  bbox_transform=ax.transAxes)  # zoom-factor: 2, location: upper-left
        axins.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0), useMathText=False)
    else:
        axins = None

    return fig, ax, axins, loc1, loc2


def plot_postprocess(fig, ax, title, experiment, dir, xlabel=r"Time $\left(\mathrm{s}\right)$", ylabel="Value",
                     unit=r"$\left(1\right)$", save=False, plot_legend=True, text_label=False):
    """
    Postprocesses the plot by setting labels, titles, the legend, and saving

    :param text_label:
    :param fig: matplotlib figure (index 0 return value of plot_init)
    :param ax: axis object (index 1 return value of plot_init)
    :param title: title of the plot
    :param experiment: experiment name (for saving)
    :param dir: direction for saving
    :param xlabel: label of the x axis
    :param ylabel: label of the y axis
    :param unit: unit of the plotted value (will be concatenated to ylabel)
    :param save: flag to indicate saving
    :param plot_legend: flag to indicate the need for the legend
    :return:
    """

    if text_label is True:
        # Note: ylabel should be textual (not Greek letters)
        ylabel = title

    ylabel = ylabel + r" " + unit

    # remove note in parenthesis
    open_idx = ylabel.find(r" (")
    if open_idx is not -1:  # only if there is a parenthesis in the label
        ylabel = ylabel[:open_idx] + ylabel[ylabel.find(r")") + 1:]

    if save is False and text_label is False:
        # Note: no titles can be included in the journal
        ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # legend = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.125),
    #                    fancybox=True, shadow=False, ncol=2)

    # plot legend
    if plot_legend is True:
        # calculate how many vectors are plot
        num_three_tuples = len(labels) // 3

        if num_three_tuples > 0:
            # if multiple, then plot the componets of each
            # in a separate column in the legend
            legend = ax.legend(loc='lower left', bbox_to_anchor=(0.625, 0.),
                               fancybox=True, shadow=False, ncol=num_three_tuples)
        else:
            legend = ax.legend()

    else:
        legend = None

    save_figure(dir, experiment, fig, legend, save, title)


def save_figure(dir, experiment, fig, legend, save, title):
    """
    Saves the figure in .png and .svg

    :param dir: directory to save in
    :param experiment: experiment name (to make the filename unique)
    :param fig: figure object
    :param legend: legend variable
    :param save: save flag
    :param title: title of the figure
    :return:
    """

    if save:
        filename = title.lower()
        filename = filename.replace(" ", "_")
        svg_path = join(dir, f"{experiment}_{filename}.svg")
        png_path = join(dir, f"{experiment}_{filename}.png")

        if legend is not None:
            fig.savefig(svg_path, bbox_extra_artists=(legend,), bbox_inches='tight',
                        format="svg", transparent=True)
            fig.savefig(png_path, bbox_extra_artists=(legend,), bbox_inches='tight',
                        format="png", transparent=True)
        else:
            fig.savefig(svg_path, format="svg", transparent=True)
            fig.savefig(png_path, format="png", transparent=True)


def format_label_unit(attr: str, error: bool = False) -> Tuple[str, str, str]:
    """
    Create the label, title, and unit strings for each attribute

    :param attr: attribute string
    :param error: is the attribute an error term
    :return: a tuple including the label, title, and unit strings for the attribute
    """

    if "omega" in attr:
        unit = r"\mathrm{\frac{rad}{s}}"

        if attr == "omega":
            label = r"\omega"
            title = "Angular velocity"
        elif attr == "omega_pred":
            label = r"\hat{\omega}"
            title = "Predicted angular velocity"

    elif "angles" in attr:
        unit = r"\mathrm{rad}"

        if attr == "angles":
            label = r"\mathrm{Euler\, angles}"
            title = "Euler angles"
        elif attr == "angles_pred":
            label = r"\mathrm{Predicted\, Euler\, angles}"
            title = "Predicted Euler angles"
        elif attr == "angles_target":
            label = r"\mathrm{Target\, Euler\, angles}"
            title = "Target Euler angles"
        elif attr == "angles_ref_pred":
            label = r"\mathrm{Predicted\, Euler\, angles (reference)}"
            title = "Predicted Euler angles (reference)"
        elif attr == "angles_esoq":
            label = r"\mathrm{ESOQ2\, Euler\, angles}"
            title = "Predicted Euler angles (ESOQ2)"
        elif attr == "angles_st":
            label = r"\mathrm{Star tracker\, Euler\, angles}"
            title = "Predicted Euler angles"

        elif attr == "ads_error_angles":
            label = r"\mathrm{ADS \, error}"
            title = "ADS error"
        elif attr == "ads_ref_error_angles":
            label = r"\mathrm{ADS \, error \, (reference)}"
            title = "ADS error"

        elif attr == "acs_error_angles":
            label = r"\mathrm{ACS \, error}"
            title = "ACS error"


    elif "mtq" in attr:

        if attr == "mtq_m":
            label = r"\mathbf{m}"
            unit = r"\mathrm{Am^{2}}"
            title = "Magnetic dipole moment of the magnetorquers"
        elif attr == "mtq_torques":
            label = r"\tau_{\mathrm{mag}}{}"
            unit = r"\mathrm{Nm}"
            title = "Actuation torques of the magnetorquers"

    elif attr == "hybrid_torques":
        label = r"\tau_{h}{}"
        unit = r"\mathrm{Nm}"
        title = "Hybrid actuation torques"

    elif "rw" in attr:
        if attr == "rw_torques":
            label = r"\tau_{\mathrm{rw}}{}"
            unit = r"\mathrm{Nm}"
            title = "Actuation torques of the reaction wheels"
        elif attr == "h_rw":
            label = r"\mathbf{L}_{\mathrm{rw}}{}"
            unit = r"\mathrm{Nms}"
            title = "Angular momentum of the reaction wheels"

    elif attr is "magnet_sun_inner_prod":
        label = r"{\langle \mathbf{s}_{ABC}; \mathbf{B}_{ABC} \rangle}"
        unit = r"1"
        title = "Inner product of the Sun vector and magnetic field vector in the ABC frame"

    elif "magnet" in attr:
        unit = r"\mathrm{T}"

        if attr == "magnet":
            label = r"\mathbf{B}"
            title = "Magnetic field vector in the ECI frame"
        elif attr == "magnet_abc":
            label = r"\mathbf{B}_{\mathrm{ABC}}{}"
            title = "Magnetic field vector in the ABC frame"

    elif "sun" in attr:
        unit = r"\mathrm{km}"

        if attr == "sun":
            label = r"\mathbf{s}"
            title = "Sun vector in the ECI frame"
        elif attr == "sun_abc":
            label = r"\mathbf{s}_{\mathrm{ABC}}{}"
            title = "Sun vector in the ABC frame"

    elif "dist_torques" in attr:
        unit = r"\mathrm{Nm}"

        if attr == "dist_torques":
            label = r"\tau_{\mathrm{dist}}{}"
            title = "Disturbance torques in the ECI frame"
        elif attr == "dist_torques_abc":
            label = r"\tau_{\mathrm{dist}}_{\mathrm{ABC}}{}"
            title = "Disturbance torques in the ABC frame"

    else:
        raise ValueError("Invalid attribute given!")

    label = r"\ensuremath{" + label + r"}"
    unit = r"\ensuremath{\left(" + unit + r"\right)}"
    title = f"{title} error" if (error and "error" not in attr) else title

    return label, unit, title


def determine_coordinate_labels(label: str, mean: np.ndarray):
    """
    Create the coordinate labels for vector plots

    :param label: label string
    :param mean: numpy array of the mean
    :return:
    """

    if mean.shape[1] is 1:  # for the magnetic field and Sun vector inner product
        labels = " "  # won't render in latex, an empty string is not OK, as it cannot be enumerated (it gives back an empty iterator)
    elif mean.shape[1] == 3:
        labels = [r"_\mathrm{x}", r"_\mathrm{y}", r"_\mathrm{z}"] if "_" not in label else [r"{}_{, \;\!\mathrm{x}}",
                                                                                            r"{}_{, \;\!\mathrm{y}}",
                                                                                            r"{}_{, \;\!\mathrm{z}}"]
        if "Euler" in label or "ADS" in label:
            labels = [r"\phi", r"\theta", r"\psi"]
            if "Predicted" in label or "ESOQ2" in label:
                labels = [r"\hat{"+l+r"}" for l in labels] 
            if "(reference)" in label:
                labels = [l+r"_{\mathrm{ref}}" for l in labels]
            if "error" in label:
                labels = [r"\Delta"+l for l in labels]
    else:
        raise ValueError("Date being plotted is neither 3D nor 1D, no labels are generated")
    return labels


class ExperimentAnalyzer(object):
    def __init__(self, experiment:str, decimate_step:int=1, log_dir:Optional[str]=None):
        """

        :param experiment: name of the experiment
        :param decimate_step: subsampling step size (every ith value will be read from the file)
        :param log_dir: root directory of the logs
        """


        # create file structure
        self.experiment_name = experiment
        self.base_dir = join(dirname(dirname(abspath(__file__))), "log" if log_dir is None else log_dir)
        self.data_dir = join(self.base_dir, self.experiment_name)

        self.fig_dir = join(self.base_dir, "figures")
        make_dir(self.fig_dir)

        self.exp_fig_dir = join(self.fig_dir, self.experiment_name)
        make_dir(self.exp_fig_dir)


        self.decimate_step = decimate_step


        # load experiments
        self.experiments = {}
        for exp in listdir(self.data_dir):
            if exp.endswith(".hdf5"):
                timestamp = exp[9:-5]
                self.experiments[timestamp] = Logger(self.experiment_name, log_dir=log_dir)
                self.experiments[timestamp].load(exp, self.decimate_step)

    def plot(self, plot_items: List[str], start_idx: int = 0, stop_idx: Optional[int] = None,
             error: Optional[int] = False, joint_plot: bool = False, plot_norm: bool = True, std_factor: int = 2,
             figsize: Tuple[int, int] = (8, 6), save: bool = False, norm_only: bool = False, plot_std: bool = True,
             plot_legend=True,
             multiplot: bool = False, multiplot_idx: Optional[List] = None, multiplot_color: Optional[str] = None,
             text_label=False):
        """
        Creates plots for analysis

        :param plot_items: list of items to plot
        :param start_idx: start index of the plot
        :param stop_idx: stop index of the plot
        :param error: flag whether to plot the error of the item
        :param joint_plot: flag whether to plot items on the same plot
        :param plot_norm: flag whether to plot the norm of the items
        :param std_factor: range of the standard deviation to plot if plot_std is True
        :param figsize: tuple of the figure size
        :param save: flag whether to save the plot (both as .png and as .svg)
        :param norm_only: flag whether to plot only the norm of the item
        :param plot_std: flag whether to plot the standard deviation (may result in large .svg files)
        :param plot_legend: flag whether to plot the legend
        :param multiplot: flag whether to plot multiple items (e.g. for a Monte Carlo run, when logs are in the same directory) and not the statistics
        :param multiplot_idx: list of indices for the multiplot case to show in the legend
        :param multiplot_color: color for the multiplot case (if given, all curves will be the same color)
        :param text_label: flag whether to use text instead of greek letters (for IEEE journals)
        :return:
        """

        # handles case if only one item shall be plotted
        if type(plot_items) is not list:
            plot_items = [plot_items]

        # extract data from the experiments
        statistics = self._extract_data(error, multiplot, plot_items, start_idx, stop_idx)

        plot_items, statistics = self._calc_magnet_sun_inner_prod(joint_plot, plot_items, statistics)

        # plot
        x_points = np.arange(list(statistics.values())[0][0].shape[0])
        # for a joint plot, initialize here
        if joint_plot:
            fig, ax, _, _, _ = plot_init(figsize, inset=False)

        for item, data in statistics.items():

            # query data
            mean, min, max, std, multidata = data

            # initialize for separate plots
            if not joint_plot:
                fig, ax, _, _, _ = plot_init(figsize, inset=False)

            # get label and unit
            label, unit, title = format_label_unit(item, error)

            # coordinate labels
            labels = determine_coordinate_labels(label, mean)


            # plot coordinate-wise
            if not norm_only and not multiplot:
                for dim, coord in enumerate(labels):

                    # append the coordinate to the label 
                    # (but not for quantities containing "ADS" or "Euler") - there we will have greek letters
                    loc_label = label + coord if ("Euler" not in label and "ADS" not in label) else coord
                    
                    # append the "err" subscript if the quantity is not indicated as an error.
                    if error is True:
                        if "error" not in label:
                            loc_label = loc_label + (
                                r"_{\mathrm{err}}" if "_" not in loc_label else r"{}_{\, ,\mathrm{err}}")
                    
                   
                    loc_label = r"\ensuremath{" + loc_label + r"}"

                    ax.plot(mean[:, dim], label=loc_label if plot_legend is True else None)
                    # ax.fill_between(x_points, max[:, dim], min[:, dim], alpha=.4)

                    if plot_std:
                        ax.fill_between(x_points, mean[:, dim] + std_factor * std[:, dim],
                                        mean[:, dim] - std_factor * std[:, dim], alpha=.4)

                # change the label for plotting error angles
                if "error" in label:
                        if "ADS" in label or "ACS" in label or "Euler" in label:                            
                            label = r"\ensuremath{\Delta\Theta}"


            # plot norm
            if plot_norm:

                if multiplot:
                    for idx, exp_data in enumerate(multidata):
                        exp_norm = np.linalg.norm(exp_data, axis=1)

                        # customize indices (e.g. for ST ablation)
                        subscript = str(idx) if multiplot_idx is None else multiplot_idx[idx]

                        multi_label = r"\ensuremath{\Vert" + label + r"_{" + subscript + r"}\Vert}"
                        ax.plot(exp_norm, color=multiplot_color, label=multi_label if plot_legend is True else None)

                else:
                    mean_norm = np.linalg.norm(mean, axis=1)
                    is_ref = "reference" in label

                    # modify the label for angle errors
                    if error is True:
                        if "error" not in label:
                            label = r"\Delta" + label #+ (r"_{\mathrm{err}}" if "_" not in label else r"{}_{\, ,\mathrm{err}}")
                        if "ADS" in label or "ACS" in label or "Euler" in label:
                            
                            label = r"\Delta\Theta"


                    # for showing variance, the plot label and the label used for the title will be different
                    if is_ref is True:
                        plot_label = label + r"_\mathrm{ref}"
                    else:
                        plot_label = label

                    # norm bars
                    label = r"\ensuremath{" + label + r"}"
                    label = r"\ensuremath{\Vert" + label + r"\Vert}"

                    plot_label = r"\ensuremath{\Vert" + plot_label + r"\Vert}"
                    
                    # std plot label
                    if plot_std:
                        plot_label = plot_label + r"+" + str(std_factor) + r"\sigma"
                        plot_label = r"\ensuremath{" + plot_label + r"}"

                    ax.plot(mean_norm, label=plot_label if plot_legend is True else None)

                    if plot_std:
                        std_norm = np.linalg.norm(std, axis=1)
                        ax.fill_between(x_points, mean_norm + std_factor * std_norm, mean_norm - std_factor * std_norm,
                                        alpha=.4)

            if not joint_plot:
                plot_postprocess(fig, ax, title, experiment=self.experiment_name, dir=self.exp_fig_dir, ylabel=label,
                                 unit=unit, save=save, plot_legend=plot_legend, text_label=text_label)

        if joint_plot:
            plot_postprocess(fig, ax, title, experiment=self.experiment_name, dir=self.exp_fig_dir, ylabel=label,
                             unit=unit, save=save, plot_legend=plot_legend, text_label=text_label)

        # return the value only if one item is plotted
        # otherwise, return the whole dict
        return statistics if len(plot_items) is not 1 else list(statistics.values())[0 if not multiplot else -1]

    def _extract_data(self, error:bool, multiplot:bool, plot_items:List[str], start_idx:Optional[int], stop_idx:Optional[int]):
        """
        Extracts data from the experiments

        :param error: flag whether to use the error of an item
        :param multiplot: flag whether to preserve all data for the given item or just statistics
        :param plot_items: list of the items to use
        :param start_idx: starting index
        :param stop_idx: stop index
        :return:
        """
        # Calculate plot statistics for each item
        statistics = {}
        for item in plot_items:
            print(item)
            data = []
            for exp in self.experiments.values():

                if not error or "error" in item:  # if the error is precalculated, no need to do it here
                    d = exp.__dict__[item].data

                else:
                    if "target" not in item:
                        d = exp.__dict__[f"{item}_pred"].data - exp.__dict__[item].data
                    else:
                        d = exp.__dict__[f"{item[:item.find('_')]}"].data - exp.__dict__[item].data

                num_points = len(d)
                data.append(d[start_idx:stop_idx if stop_idx is not None else num_points, ...])

            data = np.array(data)
            statistics[item] = (
                data.mean(axis=0),
                data.min(axis=0),
                data.max(axis=0),
                data.std(axis=0),
                data[0] if not multiplot else data  # save all data
            )
        return statistics

    @staticmethod
    def _calc_magnet_sun_inner_prod(joint_plot:bool, plot_items:List[str], statistics:Dict):
        """
        Calculate the inner product of Sun vector and magnetic field in the ABC frame

        :param joint_plot: should be True
        :param plot_items: should equal ["magnet_abc", "sun_abc"]
        :param statistics: dictionary of the statistics returned by _extract_data
        :return:
        """
        if joint_plot is True and len(
                statistics.keys()) is 2 and "magnet_abc" in statistics.keys() and "sun_abc" in statistics.keys():
            magnet_abc_mean, _, _, _, _ = statistics["magnet_abc"]
            magnet_abc_normalized_mean = magnet_abc_mean / np.linalg.norm(magnet_abc_mean, axis=1).reshape(-1, 1)
            sun_abc_mean, _, _, _, _ = statistics["sun_abc"]

            del statistics
            plot_items = ["magnet_sun_inner_prod"]
            statistics = {"magnet_sun_inner_prod": (
                (magnet_abc_normalized_mean * sun_abc_mean).sum(axis=1).reshape(-1, 1), None, None, None, None)}
        return plot_items, statistics

    def plot_detumbling_mc(self, start_idx=0, stop_idx=None, figsize=(8, 5), save=False, detumble_thresh=np.deg2rad(3)):
        """
        Plots the results of the detumbling Monte Carlo experiment
        """

        data, _, _, _, multidata = self.plot(
            # what
            ["omega"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=True, multiplot_color="tab:blue",
            # figure
            figsize=figsize,
            # flags
            error=False, joint_plot=True, plot_legend=False, text_label=True,
            # do not save here, as plot will be processed below
            save=False
        )

        """Calculate the detumbling time statistics"""
        # calc norms for each experiment
        norms = [np.linalg.norm(d, axis=1) for d in multidata]

        # select the first index with number under the detumbling threshold
        detumbling_times = np.array([np.where(d<np.deg2rad(3))[0][0] for d in norms])

        # convert to seconds with the time step argument (this is already done by decimation

        # calculate statistics
        print(f"Mean={detumbling_times.mean():.3f}s, std={detumbling_times.std():.3f}s, min={detumbling_times.min():.3f}s, max={detumbling_times.max():.3f}s")

        """Detumbling statistics - END"""

        # query data for the added plot element
        label, unit, title = format_label_unit("omega", False)

        # z order is given to plot the threshold on the top
        plt.hlines(detumble_thresh, 0, data.shape[0], zorder=100000,
                   label=r"Detumbling threshold=" + f"{detumble_thresh:.3}" + unit)
        plt.legend()

        # save figure here with additional info
        save_figure(self.exp_fig_dir, self.experiment_name, plt, None, save, title)

    def plot_omega_error(self, start_idx=0, stop_idx=None, figsize=(8, 5), save=False):
        """
        Plots the angular velocity error
        """
        w_err, _, _, _, _ = self.plot(
            # what
            ["omega"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=True, joint_plot=True, save=save, plot_std=False
        )
        w_err_stable = np.rad2deg(np.linalg.norm(w_err, axis=1))
        print(f"Mean omega error={w_err_stable.mean() * 3600} (std={w_err_stable.std() * 3600}) in arcsec/s")

    def plot_ads_error(self, start_idx=0, stop_idx=None, figsize=(8,5), save=False):
        """
        Plots the ADS error (both proposed and reference)
        """
        stat_dict = self.plot(
            # what
            ["ads_error_angles", "ads_ref_error_angles"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=True, joint_plot=True, save=save,
            # std
            std_factor=3
        )

        proposed_mean_err = np.linalg.norm(stat_dict["ads_error_angles"][0], axis=1)
        ref_mean_err = np.linalg.norm(stat_dict["ads_ref_error_angles"][0], axis=1)
        print(
            f"Mean errors: proposed={proposed_mean_err.mean() * 3600} (std={proposed_mean_err.std() * 3600}), reference={ref_mean_err.mean() * 3600} (std={ref_mean_err.std() * 3600}) in arcseconds")

    def plot_esoq_cov(self, start_idx=0, stop_idx=None, figsize=(18, 5), save=False):
        """
        Plots the ESOQ2 error dependence on the quaternion norm, and the adaptive covariance scheme
        """
        def min_max_scale(x, clip=None):
            if clip is not None:
                x = np.clip(x, -10, clip)
            x_max, x_min = x.max(), x.min()

            return (x - x_min) / (x_max - x_min)

        # constants
        error = False
        multiplot = True

        # data query
        plot_items = ["angles", "angles_esoq", "angles_st", "esoq_q_norm"]
        statistics = self._extract_data(error, multiplot, plot_items, start_idx, stop_idx)

        esoq_q_norm = statistics["esoq_q_norm"][-1][0]
        angles = statistics["angles"][-1][0]
        esoq_err = statistics["angles_esoq"][-1][0] - angles
        esoq_scaled_norm_err = min_max_scale(np.linalg.norm(esoq_err, axis=1)).reshape(-1, 1)

        # this needs a separate query due to the underlying logic
        plot_items_sun_magnet = ["magnet_abc", "sun_abc"]
        statistics_sun_magnet = self._extract_data(error, multiplot, plot_items_sun_magnet, start_idx,
                                                   stop_idx)
        joint_plot = True
        plot_items_sun_magnet, statistics_sun_magnet = self._calc_magnet_sun_inner_prod(joint_plot,
                                                                                        plot_items_sun_magnet,
                                                                                        statistics_sun_magnet)

        magnet_sun_inner_prod = statistics_sun_magnet["magnet_sun_inner_prod"][0]

        # use the same scaling as in kalman_filter.py
        inner_prod_scaled = scale_magnet_sun(magnet_sun_inner_prod)
        inv_esoq_q_norm_clipped = inv_clip_esoq_q_norm(esoq_q_norm).reshape(-1, 1)
        weight = inner_prod_scaled * inv_esoq_q_norm_clipped

        # initialize plot
        fig, ax, _, _, _ = plot_init(figsize, inset=False)

        # plot
        ax.plot(esoq_scaled_norm_err, label=r"\ensuremath{\Vert \Delta q_{\mathrm{ESOQ2}}\Vert}")
        ax.plot(inner_prod_scaled, label=r"\ensuremath{f_{\mathrm{cov}}\left(\alpha_{\mathrm{ESOQ2}}\right)}")
        # ax.plot(weight * esoq_scaled_norm_err, label=r"")

        # plot shaded background + colorbar
        cf = ax.pcolor(1 - inv_esoq_q_norm_clipped.T, cmap='gray', alpha=.4, snap=True)
        fig.colorbar(cf, ax=ax, label=r"\ensuremath{\Vert q_{\mathrm{ESOQ2}}\Vert}")

        # postprocessing
        plot_legend = True
        title = "Relationship between ESQO2 error and vector parallelism"
        plot_postprocess(fig, ax, title, experiment=self.experiment_name, dir=self.exp_fig_dir, save=save,
                         plot_legend=plot_legend)

    def plot_ads_error_box(self, start_idx=0, stop_idx=None, multiplot_idx: Optional[List] = None, figsize=(18, 5),
                           save=False):
        """
        Plots the error difference between proposed and reference methods in the ST ablation study
        """

        # constants
        error = True
        multiplot = True

        # data query
        statistics = self._extract_data(error, multiplot, ["ads_error_angles", "ads_ref_error_angles"], start_idx,
                                        stop_idx)
        proposed_data = statistics["ads_error_angles"][-1]
        ref_data = statistics["ads_ref_error_angles"][-1]

        # calculate errors
        errors = []
        for proposed, ref in zip(proposed_data, ref_data):
            errors.append(np.linalg.norm(ref, axis=1) - np.linalg.norm(proposed, axis=1))

        # to avoid cutting off parts of the plot
        rcParams.update({'figure.autolayout': True})

        # plot
        fig, ax, _, _, _ = plot_init(figsize, inset=False)

        ax.boxplot(errors, showfliers=False, labels=multiplot_idx)  # , notch=True, showmeans=True)

        label, unit, title = format_label_unit("ads_error_angles", error)

        xlabel = r"Star tracker sampling time $\left(250\ \mathrm{ms}\right)$"
        title = "Difference of the error norms of the ADS error"

        plot_postprocess(fig, ax, title, experiment=self.experiment_name, dir=self.exp_fig_dir, xlabel=xlabel,
                         ylabel=label + " difference", unit=unit, save=save, plot_legend=False)

    def plot_ads_error_mc(self, start_idx=0, stop_idx=None, multiplot_idx: Optional[List] = None,
                          figsize=(18, 5), plot_legend=True, save=False, multiplot_color: Optional[str] = None):

        """
        Plots the ADS error in the Monte Carlo study
        """
        euler_err, _, _, _, euler_multi = self.plot(
            # what
            ["ads_error_angles"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=True, multiplot_idx=multiplot_idx, multiplot_color=multiplot_color,
            # figure
            figsize=figsize,
            # flags
            error=True, joint_plot=True, plot_legend=plot_legend, save=save,
            # std
            std_factor=3
        )

        euler_err_norm = np.rad2deg(np.linalg.norm(euler_err, axis=1))
        print(
            f"Mean ADS angle error={euler_err_norm.mean() * 3600}, std={euler_err_norm.std() * 3600} in arcseconds")

        # plot errors for each configuration
        if multiplot_idx is not None:
            mean_errors = [3600 * np.linalg.norm(e, axis=1) for e in euler_multi]
            for idx, error in zip(multiplot_idx, mean_errors):
                print(f"Sampled each {idx} period with mean ADS error of {error.mean()} arcseconds (std={error.std()})")

    def plot_acs_error(self, start_idx=0, stop_idx=None, figsize=(18, 5), save=False):
        """
        Plots the ACS error
        """

        data, _, _, _, _ = self.plot(
            # what
            ["acs_error_angles"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=True, joint_plot=True, save=save,
            # std
            std_factor=3
        )

        data_norm = np.linalg.norm(data[start_idx:stop_idx], axis=1)
        print(f"Mean ACS error={data_norm.mean() * 3600} (std={data_norm.std() * 3600}) in arcseconds")

    def plot_attitude_stability(self, start_idx=0, stop_idx=None, figsize=(8, 5), save=False):
        """
        Plots the attitude and calculates its stability
        """
        euler, _, _, _, _ = self.plot(
            # what
            ["angles"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=True, norm_only=True,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=False, joint_plot=True, save=save,
            # std
            std_factor=3
        )

        # attitude stability
        att_stability = np.diff(np.rad2deg(euler), axis=0).mean(axis=0)
        print(
            f"Mean true attitude stability={att_stability.mean() * 3600} (std={att_stability.std() * 3600}) in arcseconds/s")

    def plot_separate(self, plot_items, start_idx=0, stop_idx=None, figsize=(18, 5), save=False):
        """
        Plot items in a separate plot
        """
        data, _, _, _, _ = self.plot(
            # what
            plot_items,
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=False, norm_only=False,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=False, joint_plot=False, save=save
        )

        return data

    def plot_joint(self, plot_items, start_idx=0, stop_idx=None, figsize=(18, 5), save=False):
        """
        Plot items in the same plot
        """

        data = self.plot(
            # what
            plot_items,
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=False, norm_only=False,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=False, joint_plot=True, save=save
        )

        return data

    def plot_magnet_sun_parallelism(self, start_idx=0, stop_idx=None, figsize=(18, 5), save=False):
        """
        Plot the inner product of the Sun vector and magnetic field (both in the ABC frame)
        """
        magnet_sun_inner_prod, _, _, _, _ = self.plot(
            # what
            ["sun_abc", "magnet_abc"],
            # range
            start_idx=start_idx, stop_idx=stop_idx,
            # norm
            plot_norm=False, norm_only=False,
            # multiplot
            multiplot=False,
            # figure
            figsize=figsize,
            # flags
            error=False, joint_plot=True, save=save,
            plot_std=False
        )

        return magnet_sun_inner_prod
