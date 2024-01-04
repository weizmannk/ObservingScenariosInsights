# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script calculates the geostationary altitude of the ULTRASAT mission given its Two-Line Element (TLE) set.
                  It utilizes the Skyfield library to interpret TLE data and compute the satellite's position. The script is designed
                  for robustness and provides informative output. This is particularly useful for simulations or theoretical work where a
                  fixed geostationary altitude is assumed. As the ULTRASAT mission is geostationary, a typical altitude is about 35786 km.
                  Dependencies: "skyfield". Run `pip install skyfield` to install.
---------------------------------------------------------------------------------------------------
"""

import logging
from skyfield.api import load, EarthSatellite

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def read_tle(tle_file_path):
    """
    Reads a TLE file and returns the lines.
    Args:
        tle_file_path (str): Path to the TLE file.
    Returns:
        list: Lines read from the file, or None if an error occurred.
    """
    try:
        with open(tle_file_path, "r") as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        logger.error(f"The file {tle_file_path} was not found.")
    except Exception as e:
        logger.error(f"An error occurred while reading the TLE file: {e}")
    return None


def calculate_satellite_altitude(tle_line1, tle_line2):
    """
    Calculates the altitude of a satellite given its TLE lines.
    Args:
        tle_line1 (str): The first line of the TLE.
        tle_line2 (str): The second line of the TLE.
    Returns:
        float: The altitude of the satellite in kilometers, or None if an error occurred.
    """
    try:
        satellite = EarthSatellite(tle_line1, tle_line2)
        ts = load.timescale()
        t = ts.now()
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()
        return subpoint.elevation.km
    except Exception as e:
        logger.error(f"An error occurred while calculating the altitude: {e}")
    return None


def main():
    """
    Main function to execute the script.
    """
    tle_file_path = "./configFiles/goes17.tle"
    lines = read_tle(tle_file_path)

    if lines and len(lines) >= 3:

        tle_line0 = lines[0].strip()  # Title line, typically contains the name
        tle_line1 = lines[1].strip()  # First line of actual TLE data
        tle_line2 = lines[2].strip()  # Second line of actual TLE data

        # Calculate the altitude
        altitude_km = calculate_satellite_altitude(tle_line1, tle_line2)

        if altitude_km is not None:
            altitude_m = altitude_km * 1000
            logger.info(
                f"In '{tle_line0}', ULTRASAT - Altitude is: {altitude_km:.3f} km or {altitude_m:.3f} m"
            )
        else:
            logger.error("Failed to calculate the altitude.")
    else:
        logger.error("The TLE data is invalid or incomplete.")


if __name__ == "__main__":
    main()
