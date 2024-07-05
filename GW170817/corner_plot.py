# python corner_plot.py -f ./lambda_GW170817-AT2017gfo_posterior_samples.dat  ./lambda_GW170817-AT2017gfo-GRB170817A_afterglow_posterior_samples.dat  -p  labels-axis.prior -o GW170817_AT2017gfo_GRB170817A  --kwargs "{'levels':[0.68, 0.95]}" --ext pdf  --verbose  --label-name GW170817-AT2017gfo GW170817-AT2017gfo-GRB170817A


import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.patches as mpatches
import re
import matplotlib
from ast import literal_eval
import arviz as az

matplotlib.use("agg")

params = {
    # latex
    "text.usetex": True,
    # fonts
    "mathtext.fontset": "stix",
    # figure and axes
    "axes.grid": False,
    "axes.titlesize": 10,
    # tick markers
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.size": 10.0,
    "ytick.major.size": 10.0,
    # legend
    "legend.fontsize": 20,
}
plt.rcParams.update(params)
plt.rcParams["font.serif"] = ["Computer Modern"]
plt.rcParams["font.family"] = ["serif", "STIXGeneral"]
plt.rcParams.update({"font.size": 16})


def plotting_parameters(prior_filename, filename_with_fullpath, verbose):
    """
    Extract plotting parameters and latex representation from the given prior file.
    Keys will be used as column names for the posterior samples and values will be used as axis labels.
    Parameters
    ----------
    prior_filename : str
        The file path of the prior file.
    Returns
    -------
    dict
        A dictionary containing the plotting parameters.
    """
    parameters = {}
    with open(prior_filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # ignore comments
            if line.startswith("#"):
                continue

            # checks for empty lines
            if line:
                key_value = line.split("=", 1)
                key = key_value[0].replace(" ", "")

                # ignore prior if it is a fixed value
                if re.match(r"^\s*-?[\d.]+\s*$", key_value[1]):
                    continue
                latex_label_match = re.search(
                    r"latex_label\s*=\s*(['\"])(.*?)\1", key_value[1]
                )

                # use latex label if it exists, otherwise use the name
                if latex_label_match:
                    latex_label_value = latex_label_match.group(2)
                else:
                    latex_label_value = re.search(
                        r"name\s*=\s*['\"]([^'\"]+)['\"]", key_value[1]
                    ).group(1)

                parameters[key] = latex_label_value

        for k, v in parameters.items():
            parameters[k] = v.replace("_", "-") if not v.startswith("$") else v
            parameters[k] = v.replace("\\\\", "\\") if v.startswith("$") else v

    posterior_params = set(pd.read_csv(filename_with_fullpath, sep=" ").columns)

    prior_params = set(parameters.keys())

    common_params = list(prior_params & posterior_params)

    common_params_dict = {k: parameters[k] for k in common_params}

    return common_params_dict


def load_csv(filename_with_fullpath, prior_filename, verbose):
    """
    Load posterior samples from a CSV file.
    Parameters
    ----------
    filename_with_fullpath : str
        The file path of the posterior samples in CSV format.
    prior_filename : str
        The file path of the prior file to crossmatch the parameters and only load the (astrophysical) source parameters.
    Returns
    -------
    numpy.ndarray
        A 2D numpy array representing the posterior samples.
    """
    df = pd.read_csv(filename_with_fullpath, sep=" ")
    columns = plotting_parameters(prior_filename, filename_with_fullpath, verbose)
    df = df[[col for col in columns if col in df.columns]]
    samples = np.vstack(df.values)

    if verbose:
        print(f" - {filename_with_fullpath}{20*'.'}{samples.shape}")
    return samples


def load_injection(
    prior_filename, injection_file_json, injection_num, filename_with_fullpath, verbose
):
    """
    Load injection data from a JSON file.
    Parameters
    ----------
    prior_filename : str
        The file path of the prior file.
    injection_file_json : str
        The file path of the injection JSON file.
    Returns
    -------
    numpy.ndarray
        A 1D numpy array representing the injection data to be used as truths.
    """
    df = pd.read_json(injection_file_json)
    df = df.from_records(df["injections"]["content"])
    columns = plotting_parameters(
        prior_filename, filename_with_fullpath, verbose
    ).keys()
    df = df[[col for col in columns if col in df.columns]]
    truths = np.vstack(df.iloc[injection_num].values).flatten()
    if verbose:
        print("\nLoaded Injection:")
        print(f"Truths from injection: {truths}")
    return truths


def load_bestfit(prior_filename, bestfit_file_json, filename_with_fullpath, verbose):
    """
    Load bestfit params from a JSON file.
    Parameters
    ----------
    prior_filename : str
        The file path of the prior file.
    bestfit_file_json : str
        The file path of the bestfit JSON file.
    Returns
    -------
    numpy.ndarray
        A 1D numpy array representing the bestfit params to be used as truths.
    """
    df = pd.read_json(bestfit_file_json, typ="series")
    columns = plotting_parameters(
        prior_filename, filename_with_fullpath, verbose
    ).keys()
    df = df[[col for col in columns if col in df.keys()]]
    truths = np.vstack(df.values).flatten()
    if verbose:
        print("\nLoaded Bestfit:")
        print(f"Truths from bestfit: {truths}")
    return truths


def corner_plot(data, labels, filename, truths, legendlabel, ext, verbose, **kwargs):
    color_array = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]  # example colors

    # Calculate limits for each parameter for consistent plotting
    limits = []
    for i in range(data[0].shape[1]):  # assuming all datasets have the same parameters
        combined_data = np.concatenate([dataset[:, i] for dataset in data])
        limits.append((np.min(combined_data), np.max(combined_data)))

    # Setting up the corner plot with the first dataset
    fig = corner.corner(
        data[0],
        labels=list(labels.values()),
        quantiles=[0.05, 0.95],
        title_quantiles=[[0.05, 0.5, 0.95] if len(data) == 1 else None][0],
        show_titles=[True if len(data) == 1 else False][0],
        range=limits,
        bins=40,
        truths=truths,
        color=color_array[0],
        max_n_ticks=3,
        weights=np.ones(len(data[0])) / len(data[0]),
        hist_kwargs={"density": True, "zorder": len(data)},
        contourf_kwargs={"zorder": len(data)},
        contour_kwargs={"zorder": len(data)},
        **kwargs,
    )

    # Calculate the HDI and annotate plots for each parameter
    for i, param in enumerate(labels.keys()):
        hdi_low, hdi_high = az.hdi(data[0][:, i], hdi_prob=0.9)
        ax = fig.axes[i * len(labels) + i]  # Diagonal axis
        ax.axvline(hdi_low, color="red", linestyle="--")
        ax.axvline(hdi_high, color="red", linestyle="--")
        ax.annotate(
            f"{hdi_low:.2f}",
            (hdi_low, 0.05),
            textcoords="axes fraction",
            xytext=(0, -5),
            ha="center",
            va="top",
            color="red",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red"),
        )
        ax.annotate(
            f"{hdi_high:.2f}",
            (hdi_high, 0.95),
            textcoords="axes fraction",
            xytext=(0, -5),
            ha="center",
            va="top",
            color="red",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    # Handling legends if needed
    if legendlabel:
        patches = [
            mpatches.Patch(color=color_array[i], label=legendlabel[i])
            for i in range(len(legendlabel))
        ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Save the figure
    plt.savefig(filename, format=ext, dpi=300, bbox_inches="tight")
    if verbose:
        print("\nSaved corner plot:", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate corner plot")
    parser.add_argument(
        "-f",
        "--posterior-files",
        type=str,
        nargs="+",
        required=True,
        help="CSV file path for posteriors",
    )
    parser.add_argument(
        "-p",
        "--prior-filename",
        type=str,
        required=True,
        help="Prior file path for axes labels",
    )
    parser.add_argument(
        "-l",
        "--label-name",
        type=str,
        nargs="+",
        help="Legend labels (if in latex, use '$label$') or else just use the posterior folder names",
    )
    parser.add_argument(
        "-i",
        "--injection-json",
        type=str,
        help="Injection JSON file path to be used as truth values",
    )
    parser.add_argument(
        "-n",
        "--injection-num",
        type=int,
        help=(
            "Injection number to be used as truth values, only used if injection JSON is provided; equivalent to simulation ID"
        ),
    )

    parser.add_argument(
        "--bestfit-params",
        type=str,
        help=(
            "Use the values from the bestfit_params.json file to plot the truth on the"
            " corner plot; Either use injection JSON or bestfit_params.json, not both"
        ),
    )

    parser.add_argument(
        "-e",
        "--ext",
        default="png",
        type=str,
        help="Output file extension. Default: png",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        required=True,
        type=str,
        help="output file name.",
    )
    parser.add_argument(
        "--kwargs",
        default="{}",
        type=str,
        help="kwargs to be passed to corner.corner. Eg: {'plot_datapoints': False}, enclose {} in double quotes",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Print additional information",
    )

    args = parser.parse_args()
    posterior_files = args.posterior_files
    prior_filename = args.prior_filename
    injection_json = args.injection_json
    label_name = args.label_name
    ext = args.ext
    output = args.output
    injection_num = args.injection_num
    bestfit_json = args.bestfit_params
    additional_kwargs = literal_eval(args.kwargs)
    verbose = args.verbose
    if not additional_kwargs:
        print("\nNo additional kwargs provided")

    else:
        print("\nRunning with the following additional kwargs:")
        print(
            "\t\n".join(
                f" - {key}: {value}" for key, value in additional_kwargs.items()
            )
        )

    # Generate legend labels from input file names
    legendlabel = []
    if label_name is not None:
        for i in label_name:
            legendlabel.append(i)
    else:
        legendlabel = [file for file in posterior_files]
    # Load posteriors from CSV files
    posteriors = []
    if verbose:
        print("\nPosterior Files and Shape")
    for file in posterior_files:
        posterior = load_csv(file, prior_filename, verbose)
        posteriors.append(posterior)

    if injection_json is not None:
        truths = load_injection(
            prior_filename, injection_json, injection_num, posterior_files[0], verbose
        )
    elif args.bestfit_params is not None:
        truths = load_bestfit(prior_filename, bestfit_json, posterior_files[0], verbose)
    else:
        truths = None

    labels = plotting_parameters(prior_filename, posterior_files[0], verbose)
    output_filename = output + "." + ext

    kwargs = dict(
        plot_datapoints=False,
        plot_density=False,
        plot_contours=True,
        fill_contours=True,
        label_kwargs={"fontsize": 16},
        levels=[0.68, 0.95, 0.99],  # [0.16, 0.5, 0.84],
        smooth=1,
    )

    kwargs.update(additional_kwargs)

    # the code assumes that the parameters in rest of the posterior files are the same as the first posterior file. and the prior file and posterior files have the same parameters which can be plotted

    if verbose:
        print(f"\nParameters and Axis labels ({len(labels)} common parameters):")
        for k, v in labels.items():
            print(f" - {k}: {v}")

    corner_plot(
        posteriors, labels, output_filename, truths, legendlabel, ext, verbose, **kwargs
    )

## Example usage
# python corner_plot.py -f GRB_res12_linear2dp/injection_posterior_samples.dat GRB_res12_linear4dp/injection_posterior_samples.dat -p GRB170817A_emsys_4dp.prior -o linear2d_vs_linear4dp --kwargs "{'levels':[0.05,0.5,0.95]}"
