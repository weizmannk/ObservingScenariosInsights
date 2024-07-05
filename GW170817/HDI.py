# python corner_plot.py -f ./lambda_GW170817-AT2017gfo_posterior_samples.dat  ./lambda_GW170817-AT2017gfo-GRB170817A_afterglow_posterior_samples.dat  -p  labels-axis.prior -o GW170817_AT2017gfo_GRB170817A  --kwargs "{'levels':[0.68, 0.95]}" --ext pdf  --verbose  --label-name GW170817-AT2017gfo GW170817-AT2017gfo-GRB170817A


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
    """
    df = pd.read_csv(filename_with_fullpath, sep=" ")
    columns = plotting_parameters(prior_filename, filename_with_fullpath, verbose)
    df = df[[col for col in columns if col in df.columns]]
    return df


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


def plot_hdi(data, labels, output_filename):
    """Plot histograms with HDIs for each parameter."""
    fig, axes = plt.subplots(
        len(labels), 1, figsize=(10, 5 * len(labels)), squeeze=False
    )
    axes = axes.flatten()

    for i, (param, label) in enumerate(labels.items()):
        ax = axes[i]
        param_data = data[param].dropna().to_numpy()  # Convert to numpy array
        hdi = az.hdi(param_data, hdi_prob=0.95)  # Compute HDI on numpy array

        # Plot histogram
        ax.hist(param_data, bins=30, density=True, color="skyblue", alpha=0.7)
        ax.axvline(
            x=hdi[0], color="red", linestyle="--", label=f"95% HDI Lower: {hdi[0]:.2f}"
        )
        ax.axvline(
            x=hdi[1],
            color="green",
            linestyle="--",
            label=f"95% HDI Upper: {hdi[1]:.2f}",
        )
        ax.set_title(f"{label} - Histogram and 95% HDI")
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for HDI")
    parser.add_argument(
        "-f",
        "--posterior-files",
        type=str,
        nargs="+",
        required=True,
        help="CSV file path for posteriors",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output file name for the plot"
    )
    parser.add_argument(
        "-p",
        "--prior-filename",
        type=str,
        required=True,
        help="Prior file path for axes labels",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Print additional information",
    )
    args = parser.parse_args()

    verbose = args.verbose
    prior_filename = args.prior_filename
    posteriors = [
        load_csv(file, prior_filename, verbose) for file in args.posterior_files
    ]
    data = pd.concat(posteriors, ignore_index=True)
    labels = plotting_parameters(prior_filename, args.posterior_files[0], verbose)

    plot_hdi(data, labels, args.output)


fig = corner.corner(
    data[0],
    labels=list(labels.values()),
    quantiles=[0.05, 0.95],  # [0.16, 0.84],
    title_quantiles=[[0.05, 0.5, 0.95] if len(data) == 1 else None][
        0
    ],  # [0.16, 0.5, 0.84]
    show_titles=[True if len(data) == 1 else False][0],
    range=limit,
    bins=40,
    truths=truth_values,
    color=color_array[0],
    max_n_ticks=3,
    weights=np.ones(len(data[0])) / len(data[0]),
    hist_kwargs={"density": True, "zorder": len(data)},
    contourf_kwargs={
        "zorder": len(data),
    },
    contour_kwargs={"zorder": len(data)},
    **kwargs,
)
