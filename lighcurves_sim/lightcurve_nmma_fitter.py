"""
Step 1: Download SVD Files from GitLab

To begin, we need to download the SVD files essential for our analysis.
Note that for the ULTRASAT filter, TensorFlow has been specifically trained. Execute the following command to download these files:

python -m nmma.utils.models --model=Bu2019lm_tf  --filters=ultrasat --svd-path='./svdmodels' --source='gitlab'

Step 2: Create JSON Injection File

nmma-create-injection --prior-file priors/Bu2019lm.prior --injection-file ./bns_O5_injections.dat --eos-file ./example_files/eos/ALF2.dat --binary-type BNS --original-parameters --extension json -f ./outdir/injection_Bu2019lm --generation-seed 42 --aligned-spin

Step 3: Light Curve Production

lightcurve-analysis --model Bu2019lm --svd-path ./svdmodels --interpolation-type sklearn_gp --outdir ./outdir/BNS/0 --label injection_Bu2019lm_0 --prior priors/Bu2019lm.prior --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 1024 --Ebv-max 0 --injection ./outdir/injection_Bu2019lm.json --injection-num 0 --injection-detection-limit 23.5 --injection-outfile ./outdir/BNS/0/lc.csv --generation-seed 42 --filters ultrasat --plot --remove-nondetections --local-only

"""
import subprocess
import numpy as np
import os
import pandas as pd
import json


def create_injection(
    prior_file,
    injection_file,
    eos_file,
    binary_type,
    output_directory,
    generation_seed,
    extension="json",
    aligned_spin=True,
):
    command = [
        "nmma-create-injection",
        "--prior-file",
        prior_file,
        "--injection-file",
        injection_file,
        "--eos-file",
        eos_file,
        "--binary-type",
        binary_type,
        "--original-parameters",
        "--extension",
        extension,
        "-f",
        output_directory,
        "--generation-seed",
        str(generation_seed),
    ]

    if aligned_spin:
        command.append("--aligned-spin")

    print("Executing command:", " ".join(command))

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())

    if process.poll() == 0:
        print("Injection created successfully.")
    else:
        print("Error in creating injection.")


def download_svd_model_files():
    command = [
        "python",
        "-m",
        "nmma.utils.models",
        "--model",
        "Bu2019lm_tf",
        "--filters",
        "ultrasat",
        "--svd-path",
        "./svdmodels",
        "--source",
        "gitlab",
    ]

    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if process.returncode == 0:
        print("SVD model files downloaded successfully.")
    else:
        print("Error downloading SVD model files:", process.stderr)


def create_condor_sh(
    json_inject_file,
    light_curve_analysis,
    model,
    prior_file,
    output_directory,
    binary_type,
    filters_str,
    svd_path,
    generation_seed,
    local_only,
):
    with open(json_inject_file, "r") as f:
        injection_data = json.load(f)
        datadict = injection_data["injections"]["content"]
        dataframe_from_inj = pd.DataFrame.from_dict(datadict)

    if dataframe_from_inj.empty:
        return

    injection_num = len(dataframe_from_inj)
    with open("condor.sh", "w") as fid:
        for ii in range(injection_num):
            inject_outdir = os.path.join(output_directory, f"{binary_type}/{ii}")
            os.makedirs(inject_outdir, exist_ok=True)
            fid.write(
                f"{light_curve_analysis} --model {model[binary_type]} --svd-path {svd_path} --interpolation-type tensorflow --outdir {inject_outdir} --label {model[binary_type]}_{ii} --prior {prior_file} --tmin 0 --tmax 20 --dt 0.5 --error-budget 1 --nlive 1024 --Ebv-max 0 --injection {json_inject_file} --injection-num {ii} --injection-detection-limit 23.5 --injection-outfile {inject_outdir}/lc.csv --generation-seed {generation_seed} --filters {filters_str} --plot --remove-nondetections{' --local-only' if local_only else ''}\n"
            )


def main():
    output_directory = "./outdir"
    os.makedirs(output_directory, exist_ok=True)

    binary_type = "BNS"
    model = {"BNS": "Bu2019lm", "NSBH": "Bu2019nsbh"}
    prior = {"BNS": "Bu2019lm.prior", "NSBH": "Bu2019nsbh.prior"}
    prior_file = f"priors/{prior[binary_type]}"
    label = f"injection_{model[binary_type]}"

    obs_injection_file = "./bns_O5_injections.dat"
    eos_file = "./example_files/eos/ALF2.dat"
    extension = "json"

    create_injection(
        prior_file=prior_file,
        injection_file=obs_injection_file,
        eos_file=eos_file,
        binary_type=binary_type,
        output_directory=f"{output_directory}/{label}",
        generation_seed=42,
        extension=extension,
        aligned_spin=True,
    )

    download_svd_model_files()

    json_inject_file = f"{output_directory}/{label}.json"
    light_curve_analysis = (
        subprocess.check_output(["which", "lightcurve-analysis"]).decode().strip()
    )
    filters_str = "ultrasat"
    svd_path = "./svdmodels"
    generation_seed = 42
    local_only = True

    create_condor_sh(
        json_inject_file,
        light_curve_analysis,
        model,
        prior_file,
        output_directory,
        binary_type,
        filters_str,
        svd_path,
        generation_seed,
        local_only,
    )


if __name__ == "__main__":
    main()
