import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_lamda_tilde(m1, m2, lambda_1, lambda_2):
    """
    Calculate the reduced tidal deformability.

    Parameters:
    - m1: Mass of the first object.
    - m2: Mass of the second object.
    - lambda_1: Tidal deformability of the first object.
    - lambda_2: Tidal deformability of the second object.

    Returns:
    - The reduced tidal deformability.
    """
    numerator = (16 / 13) * (
        (m1 + 12 * m2) * m1**4 * lambda_1 + (m2 + 12 * m1) * m2**4 * lambda_2
    )
    denominator = (m1 + m2) ** 5
    return numerator / denominator


def process_file(file_path):
    """
    Process a file to calculate and add the reduced tidal deformability.

    Parameters:
    - file_path: Path to the data file.
    """
    try:

        df = pd.read_csv(file_path, sep=" ")
        df["lamda_tilde"] = df.apply(
            lambda row: calculate_lamda_tilde(
                row["mass_1"], row["mass_2"], row["lambda_1"], row["lambda_2"]
            ),
            axis=1,
        )

        filename = file_path.split("/")[-1]

        output_file_path = f"lambda_{filename}"
        df.to_csv(output_file_path, sep=" ", index=False)
        logging.info(f"Updated data saved to {output_file_path}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")


def main():
    """
    Main function to process multiple data files.
    """
    file_paths = [
        "./posterior_samples/GW170817-AT2017gfo_posterior_samples.dat",
        "./posterior_samples/GW170817-AT2017gfo-GRB170817A_afterglow_posterior_samples.dat",
    ]

    for file_path in file_paths:
        process_file(file_path)


if __name__ == "__main__":
    main()
