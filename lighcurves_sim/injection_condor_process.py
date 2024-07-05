import subprocess
import os
import json
import pandas as pd


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
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()

    print(stdout)
    if stderr or process.returncode != 0:
        raise Exception(
            f"Error in creating injection. Return code: {process.returncode} Error: {stderr}"
        )


output_directory = "OUTPUT"
os.makedirs(output_directory, exist_ok=True)
log_dir = os.path.join(output_directory, "logs")
os.makedirs(log_dir, exist_ok=True)

binary_type = "BNS"
model = "Bu2019lm"
prior_file = f"./nmma/priors/Bu2019lm.prior"
eos_file = "./nmma/example_files/eos/ALF2.dat"
generation_seed = 42
label = f"injection_{model}"
json_inject_file = os.path.join(output_directory, f"{label}.json")

obs_injection_file = "./bns_injections.dat"

generation_seed = 42

# create_injection(
#     prior_file=prior_file,
#     injection_file=obs_injection_file,
#     eos_file=eos_file,
#     binary_type=binary_type,
#     output_directory=output_directory,
#     generation_seed=generation_seed,
#     extension="json",
#     aligned_spin=True
# )

# Load the generated injection JSON data
try:
    with open(json_inject_file, "r") as f:
        injection_data = json.load(f)
except IOError:
    raise Exception("Failed to load injection JSON file.")

dataframe_from_inj = pd.DataFrame(injection_data["injections"]["content"])

light_curve_analysis_cmd = (
    subprocess.check_output(["which", "lightcurve-analysis"]).decode().strip()
)
interpolation_type = "sklearn_gp"
svd_path = "/home/weizmann.kiendrebeogo/ULTRASAT/SVD-MODELS/nmma-models/models"
filters_str = "sdssu,ps1__g,ps1__r,ps1__i,ps1__y,ps1__z"
detection_limit_lsst = "23.9,25.0,24.7,24.0,23.3,22.1"


job = 0
for ii, row in dataframe_from_inj.iterrows():
    inject_outdir = os.path.join(output_directory, f"no-skymap/{ii}")
    os.makedirs(inject_outdir, exist_ok=True)

    arguments = (
        f"--model {model} --svd-path {svd_path} --interpolation-type {interpolation_type} "
        f"--outdir {inject_outdir} --label {label}_{ii} --prior {prior_file} "
        f"--tmin 0 --tmax 20 --dt 0.5 --error-budget 0.1 --nlive 2048 --Ebv-max 0 "
        f"--injection {json_inject_file} --injection-num {ii} "
        f"--injection-detection-limit {detection_limit_lsst} --injection-outfile {inject_outdir}/lc.csv "
        f"--generation-seed {generation_seed} --filters {filters_str} --plot --remove-nondetections --local-only"
    )

    condor_submit_script = f"""
                                universe = vanilla
                                accounting_group = ligo.dev.o4.cbc.pe.bayestar
                                getenv = true
                                on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)
                                on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)
                                on_exit_hold_reason = (ExitBySignal == True ? "The job exited with signal " . ExitSignal : "The job exited with code " . ExitCode)
                                executable = {light_curve_analysis_cmd}
                                arguments = {arguments}
                                output = {log_dir}/$(Cluster)_$(Process).out
                                error = {log_dir}/$(Cluster)_$(Process).err
                                log = {log_dir}/$(Cluster)_$(Process).log
                                request_memory = 8192 MB
                                request_disk = 2000 MB
                                JobBatchName = NMMA
                                environment = "OMP_NUM_THREADS=1"
                                queue
                            """

    with subprocess.Popen(
        ["condor_submit"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as proc:
        stdout, stderr = proc.communicate(input=condor_submit_script)

    if stdout:
        print("Condor submit output:", stdout)
    if stderr or proc.returncode != 0:
        print("Condor submit error:", stderr)

    job += 1


print(f"\n\n{job} jobs have been submit.")
