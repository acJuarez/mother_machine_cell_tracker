import argparse
import json
import subprocess
import sys
from pathlib import Path

"""
Submit one sbatch job per (experiment, FOV, peak_id) combination to run track_astra in parallel 
should help save time

args are as follows: 
    --base-path path where experiments are located  \ 
    --time-dict JSON string in the following format'{"Exp_name":{"000":{"1343":{"start":62,"end":null}}}}' \
    [--batch-script Slurm/track_astra.batch] \

With --dry-run the sbatch commands are printed but not executed.
"""

def submit_jobs(
    base_path: str,
    time_dict_str: str,
    batch_script: str,
    notify_batch: str,
    email: str,
) -> None:
    time_dict = json.loads(time_dict_str)

    jobs = []
    for exp_name, fov_dict in time_dict.items():
        for fov_id, peak_dict in fov_dict.items():
            for peak_id, time_info in peak_dict.items():
                jobs.append((exp_name, fov_id, peak_id, time_info))

    print(f"submitting {len(jobs)} job(s).\n")

    job_ids = []
    for exp_name, fov_id, peak_id, time_info in jobs:
        # Build a single-entry JSON so each job only processes one trench
        single_dict = {exp_name: {fov_id: {peak_id: time_info}}}
        single_json = json.dumps(single_dict)

        job_name = f"trackastra_{exp_name}_{fov_id}_{peak_id}"

        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            batch_script,
            base_path,
            single_json,
        ]

        print(f"experiment: {job_name}")
        print(f" time_range: {time_info}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"-- {result.stdout.strip()}")
            # sbatch prints "Submitted batch job <ID>"; grab the trailing ID.
            job_ids.append(result.stdout.strip().split()[-1])
        else:
            print(f"ERROR: {result.stderr.strip()}", file=sys.stderr)

    submit_notifier(job_ids, notify_batch, email)


def submit_notifier(job_ids: list, notify_batch: str, email: str) -> None:
    """Submit one summary-email job that runs after all experiment jobs finish.

    Uses afterany so it fires once every experiment job completes regardless of
    success/failure, allowing failures to be reported in the single summary email.
    """
    if not job_ids:
        print("\nNo jobs were submitted; skipping summary notifier.", file=sys.stderr)
        return

    dependency = "afterany:" + ":".join(job_ids)
    cmd = [
        "sbatch",
        f"--dependency={dependency}",
        notify_batch,
        " ".join(job_ids),
        email,
    ]

    print(f"\nsubmitting summary notifier for {len(job_ids)} job(s).")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"-- {result.stdout.strip()}")
    else:
        print(f"ERROR submitting notifier: {result.stderr.strip()}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit one sbatch trackastra job per (exp, FOV, peak_id) combination"
    )
    parser.add_argument(
        "--base-path",
        required=True,
        help="Path to the directory containing all microscopy experiments",
    )
    parser.add_argument(
        "--time-dict",
        required=True,
        help='JSON time-range dict: {"Exp":{"FOV":{"Peak":{"start":N,"end":N}}}}',
    )
    parser.add_argument(
        "--batch-script",
        default="/home/users/aj0204/repos/mother_machine_cell_tracker/Slurm/track_astra.batch",
        help="Path to the SLURM batch script (default: Slurm/track_astra.batch)",
    )
    parser.add_argument(
        "--notify-batch",
        default="/home/users/aj0204/repos/mother_machine_cell_tracker/Slurm/trackastra_notify.batch",
        help="Path to the summary-email SLURM batch script (default: Slurm/trackastra_notify.batch)",
    )
    parser.add_argument(
        "--email",
        default="aj0204@stanford.edu",
        help="Recipient for the single summary email (default: aj0204@stanford.edu)",
    )

    args = parser.parse_args()
    submit_jobs(
        args.base_path,
        args.time_dict,
        args.batch_script,
        args.notify_batch,
        args.email,
    )
