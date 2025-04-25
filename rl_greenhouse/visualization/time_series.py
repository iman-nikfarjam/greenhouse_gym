from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from rl_greenhouse.greenhouse.types import Sequence


def compare_sequences(
        sequences: List[Sequence],
        labels: Optional[List[str]] = None,
        variables: List[str] = None,
        line_styles: List[str] = None,
        show: bool = True,
        labels_to_squash_factor: Dict[str, float] = None,
        labels_to_single_line: List[str] = None,
        labels_to_group: Dict[str, List[str]] = None,
        length_fraction: float = 1,
        legend_on_top: bool = False,
        **kwargs
):
    """
    Compares two full states

    :param sequences:
    :param labels:
    :param variables:
    :param line_styles:
    :param show:
    :param labels_to_squash_factor:
    :param labels_to_single_line:
    :param labels_to_group:
    """
    if variables and "time" not in variables:
        variables = ["time"] + variables

    max_len = 0
    np_arrays = []
    idx_sequence_with_max_len = 0
    for idx, sequence in enumerate(sequences):
        # # We show everything except for the resulting state as there is no action defined for the final time step.
        extra_dim_array = np.expand_dims(sequence.to_numpy(normalized=False, variables=variables), axis=0)
        np_arrays.append(extra_dim_array)
        possible_new_max_len = extra_dim_array.shape[1]
        if possible_new_max_len > max_len:
            idx_sequence_with_max_len = idx
        max_len = max(max_len, possible_new_max_len)

    nr_arrays_length_corrected = []
    for array in np_arrays:
        len_dif = max_len - array.shape[1]
        if len_dif != 0:
            added_section = np.ones((1, len_dif, array.shape[2]))
            added_section[:, :, 1:] *= float('nan')
            array = np.concatenate([array, added_section], axis=1)

        nr_arrays_length_corrected.append(array)
    all_states = np.concatenate(nr_arrays_length_corrected, axis=0)
    plot_history_3d(
        all_states,
        labels_y=sequences[0].labels(variables),
        labels_z=labels,
        line_styles=line_styles,
        index_with_max_len=idx_sequence_with_max_len,
        show=show,
        labels_to_squash_factor=labels_to_squash_factor,
        labels_to_single_line=labels_to_single_line,
        labels_to_group=labels_to_group,
        length_fraction=length_fraction,
        legend_on_top=legend_on_top,
        **kwargs
    )


def plot_history(
        history: np.array,
        labels: Optional[List[str]] = None,
        share_x: bool = False,
        save_location: str = None,
        show: bool = True,
        improve_labels: bool = True,
) -> None:
    """
    Basic plot function for any history. Plots a (n_samples, n_observation_size) shaped array in separate plots.
    The first column is placed on the x-axis. All other columns are placed in separate plots on the y-axis.

    :param history: Full history with shape (n_samples, n_observation_size).
    :param labels: List of labels with size n_observation_size
    :param share_x: If True, plot only x-ticks on the bottom figure, False to plot tick labels for x on every plot.
    :param save_location: To which name to save the file to. Defaults to None.
    :param show: True to render results.
    :param improve_labels: True to replace subplot labels with fancy labels if they exist. Defaults to True.
    """

    if len(history.shape) != 2:
        raise ValueError("plot_history can only plot arrays with a 2 dimensional shape")

    n_vars = history.shape[1] - 1  # First variable is for the x-axis.
    fig, axs = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars), sharex=share_x)

    for i in range(1, n_vars + 1):
        ax = axs[i - 1]
        ax.plot(history[:, 0], history[:, i])
        ax.grid()
        if labels:
            ax.set_xlabel(labels[0])
            y_label = IMPROVED_LABELS.get(labels[i], labels[i]) if improve_labels else labels[i]
            ax.set_ylabel(y_label)

    plt.tight_layout()
    if save_location:
        plt.savefig(save_location)
    if show:
        plt.show()


def plot_history_3d(
        history: np.array,
        labels_y: Optional[List[str]] = None,
        labels_z: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        share_x: bool = False,
        save_location: str = None,
        show: bool = False,
        improve_labels: bool = True,
        line_styles: List[str] = None,
        index_with_max_len: int = 0,
        labels_to_squash_factor: Dict[str, float] = None,
        labels_to_single_line: List[str] = None,
        labels_to_group: Dict[str, List[str]] = None,
        length_fraction: float = 1,
        legend_on_top: bool = False,
) -> None:
    """
    Basic plot function for any history. Plots a (n_samples, n_observation_size, n_categories) shaped array in separate
    plots. The first column is placed on the x-axis. All other columns are placed in separate plots on the y-axis.

    :param history: Full history with shape (n_categories, n_samples, n_observation_size).
    :param labels_y: List of labels with size n_observation_size
    :param labels_z: List of category names with size n_categories
    :param colors:
    :param share_x: If True, plot only x-ticks on the bottom figure, False to plot tick labels for x on every plot.
    :param save_location: To which name to save the file to. Defaults to None.
    :param show: True to render results.
    :param improve_labels:
    :param line_styles:
    :param index_with_max_len:
    :param labels_to_squash_factor: Squash key label to factor value times smaller.
    :param labels_to_single_line: List of labels to plot a single line for. A squashed plot only contains the first
        plot entry. Legend is omitted.
    :param labels_to_group: Dict of group name to label groups (dict with str -> list[str]). Each group will be plotted
        where the vertical label is equal to the group name. All group entries will only plot the first category line.
        This means that for each group there will be lines equal to the group size. Legend shows the names of the
        grouped variables.
    """

    if labels_y is None and (labels_to_single_line or labels_to_group):
        raise AssertionError("Set labels_y when using squashing or grouping in a timeseries plot.")

    if len(history.shape) != 3:
        raise ValueError("plot_history_3d can only plot arrays with a 3 dimensional shape")

    label_to_index = {label: labels_y.index(label) for label in labels_y}
    if labels_to_group:
        all_labels_in_groups = []
        for label_group in labels_to_group.values():
            for label in label_group:
                if label not in all_labels_in_groups:
                    all_labels_in_groups.append(label)

        indices_to_skip = [labels_y.index(label) for label in all_labels_in_groups]
    else:
        all_labels_in_groups = []
        indices_to_skip = []

    n_vars_single = history.shape[2] - 1 - len(all_labels_in_groups)  # First variable is for the x-axis.
    n_vars_grouped = len(labels_to_group) if labels_to_group else 0
    n_vars = n_vars_single + n_vars_grouped
    n_categories = history.shape[0]
    all_color = f"C{n_categories}"

    if labels_to_squash_factor is None:
        labels_to_squash_factor = {}

    height_ratios = []
    for col_idx in range(1, n_vars + 1):
        if labels_to_group and col_idx in indices_to_skip:
            continue
        height_ratios.append(1 / labels_to_squash_factor.get(labels_y[col_idx], 1))
    if labels_to_group:
        for group_name in labels_to_group.keys():
            height_ratios.append(1 / labels_to_squash_factor.get(group_name, 1))

    length_reduction = 0.5 + 0.5 * sum(height_ratios) / len(height_ratios)
    fig, axs = plt.subplots(n_vars, 1, figsize=(12, 2 * n_vars * length_reduction * length_fraction), sharex=share_x,
                            gridspec_kw={'height_ratios': height_ratios})

    nr_points = history.shape[1]
    day_mode = nr_points > 48

    skipped_plots = 0
    for col_idx in range(1, n_vars + 1):

        # We skip regular plotting if it is plotting in a group.
        if labels_to_group and col_idx in indices_to_skip:
            skipped_plots += 1
            continue

        ax = axs[col_idx - 1 - skipped_plots]
        if labels_y:
            if labels_to_single_line:
                squash_this_plot = labels_y[col_idx] in labels_to_single_line
            else:
                squash_this_plot = False
            y_label = IMPROVED_LABELS.get(labels_y[col_idx], labels_y[col_idx]) if improve_labels else labels_y[col_idx]
        else:
            y_label = None
            squash_this_plot = False

        for cat_idx in range(n_categories):
            color = (colors[cat_idx] if colors else None) if not squash_this_plot else all_color
            x_data = history[cat_idx, :, 0]
            if day_mode:
                x_data = x_data / 24

            ls = '-' if line_styles is None else line_styles[cat_idx]

            y_data = history[cat_idx, :, col_idx]
            if squash_this_plot:
                plot_label = "All"
            else:
                plot_label = str(cat_idx) if labels_z is None else labels_z[cat_idx]

            ax.plot(x_data, y_data, color=color, linewidth=1, ls=ls, label=plot_label)
            if squash_this_plot:
                break

        if y_label:
            ax.set_ylabel(y_label)

    # Grouped Plots
    if labels_to_group:
        for ax, (group_name, group_labels) in zip(axs[n_vars_single:], labels_to_group.items()):
            for label in group_labels:
                col_index = label_to_index[label]

                x_data = history[0, :, 0]
                if day_mode:
                    x_data = x_data / 24

                y_data = history[0, :, col_index]
                ax.plot(
                    x_data, y_data, linewidth=1, ls='-', color=all_color,
                    label=IMPROVED_LABELS.get(label, label) if improve_labels else label,

                )
            ax.set_ylabel(group_name)

    if legend_on_top:
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.60), ncol=n_categories)

    # All plots
    for ax in axs:
        if not legend_on_top:
            ax.legend(loc="upper right")
        ax.set_xlabel('Time [d]' if day_mode else "Time [h]")

        all_x_values = history[index_with_max_len, :, 0]
        if day_mode:
            tick_every_n_days = int(len(all_x_values) / (24 * 40)) + 1
            day_ticks = [0] + [int(val / 24) for val in all_x_values if val % 24 == 0]
            labels = [str(tick) if (tick % tick_every_n_days) == 0 else "" for tick in day_ticks]
            ax.set_xticks(day_ticks, labels=labels)
        else:
            ax.set_xticks([val for val in all_x_values])

        ax.grid()

    plt.tight_layout()
    if save_location:
        plt.savefig(save_location)
    if show:
        plt.show()


IMPROVED_LABELS = {
    "time": "Time [h]",
    "hour": "Hour [h]",
    "cum_electricity_usage": "Elec Used [kWh/m²]",
    "cum_heat_usage": "Heat Used [MJ/m²]",
    "cum_carbon_usage": "CO₂ Used [kg/m²]",
    "price_electricity": "Price Elec [€/kWh]",
    "price_heat": "Price Heat [€/MJ]",
    "price_carbon": "Price CO₂ [€/kg]",
    "costs_var_electricity": "Elec Var Cost [€/m²]",
    "costs_var_heat": "Heat Var Cost [€/m²]",
    "costs_var_carbon": "CO₂ Var Cost [€/m²]",
    "average_head_per_m2": "Avg Plant D [1/m²]",
    "gains_plant": "Plant Sales [€/m²]",
    "costs_var_total": "Sum Var Costs [€/m²]",
    "costs_fix_total": "Sum Fix Costs [€/m²]",
    "balance": "Balance [€/m²]",
    "mass": "Fresh Weight [g]",
    "quality": "Plant Quality [-]",
    "plants_per_m2": "Plant Density [1/m²]",
    "temperature_air": "Air Temp In [°C]",
    "relative_humidity": "Rel Humid In [-]",
    "par_light": "PAR In [µmol/m²s]",
    "carbon_ppm": "CO₂ In [PPMv]",
    "heating_pipe_temperature": "Pipe Temp [°C]",
    "electric_power": "Elec Power [W/m²]",
    "illumination_out": "Illumin Out [W/m²]",
    "relative_humidity_out": "Rel Humid Out [-]",
    "temperature_out": "Temp Out [°C]",
    "wind_out": "Wind Speed [m/s]",
    # "heater_power": "Heater Power [W]",
    # "co2_flow_rate": "CO₂ Flow [kg/m²s]",
    "heater_power": "Heater Power [-]",
    "co2_flow_rate": "CO₂ Flow Rate [-]",
    "ventilation_position": "Vent Pos [-]",
    "screen_blackout_position": "Black Scr Pos [-]",
    "screen_transparent_position": "Trans Scr Pos [-]",
    "enable_lamps": "Lamps [-]",
}

if __name__ == "__main__":
    for key, val in IMPROVED_LABELS.items():
        if len(val) > 20:
            print(val.ljust(50), len(val))
