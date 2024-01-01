# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script calculates the necessary HEALPix Nside parameter for a desired
                  sky coverage area in square degrees, generates sky coordinates for each pixel,
                  and saves the tessellation to a file. The script can be run from the command line
                  with the desired area and telescope name as arguments. The 'ceil_pow_2' function
                  is used to ensure the nside is a power of 2, which is a requirement for some
                  HEALPix functions.
                  Dependencies: astropy, healpy, numpy, math
Usage           : python skygrid_generation.py --area <area_in_deg2> --telescope <telescope_name>
Example         : python skygrid_generation.py --area 204 --telescope ULTRASAT
---------------------------------------------------------------------------------------------------
"""

# Or simply install dorado-scheduling , https://github.com/nasa/dorado-scheduling
# pip install dorado-scheduling
# then run : dorado-scheduling-skygrid --area "204 deg2" --output  ULTRASAT.tess --method  healpix

import argparse
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import math


def ceil_pow_2(n):
    """Return the least integer power of 2 that is greater than or equal to n."""
    mantissa, exponent = math.frexp(n)
    return math.ldexp(
        1 if mantissa >= 0 else float("nan"),
        exponent - 1 if mantissa == 0.5 else exponent,
    )


def calculate_nside(desired_area):
    """Calculate the appropriate nside for the desired area coverage."""
    nside = np.sqrt(4 * np.pi / (desired_area / (180 / np.pi) ** 2) / 12)
    nside = int(max(ceil_pow_2(nside), 1))
    return nside


def generate_sky_coords(nside):
    """Generate sky coordinates for each pixel in the ICRS frame."""
    npix = hp.nside2npix(nside)
    return [
        SkyCoord(
            ra=hp.pix2ang(nside, i, lonlat=True)[0] * u.degree,
            dec=hp.pix2ang(nside, i, lonlat=True)[1] * u.degree,
            frame="icrs",
        )
        for i in range(npix)
    ]


def save_tessellation(sky_coords, filename):
    """Save the tessellation to a file."""
    try:
        with open(filename, "w") as file:
            for i, coord in enumerate(sky_coords, start=1):
                file.write(f"{i} {coord.ra.deg} {coord.dec.deg}\n")
    except IOError as e:
        print(f"Error saving file: {e}")


def main():
    parser = argparse.ArgumentParser(description="HEALPix Tessellation Generator")
    parser.add_argument(
        "--area", type=float, required=True, help="Desired area in square degrees"
    )
    parser.add_argument(
        "--telescope", type=str, default="telescope", help="Telescope name"
    )
    args = parser.parse_args()

    desired_area = args.area
    telescope = args.telescope
    nside = calculate_nside(desired_area)
    print(f"Approximate nside for {telescope}: {nside}")

    sky_coords = generate_sky_coords(nside)
    save_tessellation(sky_coords, f"{telescope}.tess")


if __name__ == "__main__":
    main()
