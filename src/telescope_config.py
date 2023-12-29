def generate_config_file(file_path, telescope_params):
    config_data = f"""filt {telescope_params['filter']}
magnitude {telescope_params['magnitude']}
exposuretime {telescope_params['exposure_time']}
latitude {telescope_params['latitude']}
longitude {telescope_params['longitude']}
elevation {telescope_params['elevation']}
FOV_coverage {telescope_params['FOV_coverage']}
FOV {telescope_params['FOV']}
FOV_coverage_type {telescope_params['FOV_coverage_type']}
FOV_type {telescope_params['FOV_type']}
tesselationFile {telescope_params['tesselation_file']}
slew_rate {telescope_params['slew_rate']}
readout {telescope_params['readout']}
horizon {telescope_params['horizon']}
overhead_per_exposure {telescope_params['overhead_per_exposure']}
min_observability_duration {telescope_params['min_observability_duration']}
filt_change_time {telescope_params['filt_change_time']}
"""

    with open(file_path, "w") as config_file:
        config_file.write(config_data.strip())


# Specify the file path where you want to create the config file
file_path = "../ultrasat/ULTRASAT.config"

# Define telescope-specific parameters in a dictionary
telescope_params = {
    "filter": "uvot__uvm2",
    "magnitude": 22.5,
    "exposure_time": 900,
    "latitude": 0,
    "longitude": 0,
    "elevation": 0,
    "FOV_coverage": 14.28,
    "FOV": 14.28,
    "FOV_coverage_type": "square",
    "FOV_type": "square",
    "tesselation_file": "../input/ULTRASAT.tess",
    "slew_rate": 0.5,
    "readout": 0,
    "horizon": 0,
    "overhead_per_exposure": 0.0,
    "min_observability_duration": 0.0,
    "filt_change_time": 0.0,
}

# Generate the config file using the specified parameters
generate_config_file(file_path, telescope_params)
