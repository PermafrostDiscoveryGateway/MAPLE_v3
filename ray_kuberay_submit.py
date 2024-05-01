import argparse
import logging
import time

from ray.job_submission import JobSubmissionClient, JobStatus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch Ray job using JobSubmissionClient"
    )

    # Required Arguments
    parser.add_argument(
        "--host",
        required=True,
        default="http://localhost:8000/",
        help="Clowder host",
    )

    parser.add_argument(
        "--dataset_id",
        required=True,
        help="Clowder dataset from which to get input files and model weights",
    )

    parser.add_argument(
        "--key",
        required=True,
        help="Clowder API key",
    )

    parser.add_argument(
        "--env",
        required=True,
        default="environment_maple.yml",
        help="Conda yml environment",
    )
    args = parser.parse_args()

    client = JobSubmissionClient("http://127.0.0.1:8265")
    job_id = client.submit_job(
        # Entrypoint shell command to execute
        entrypoint=f"python ray_maple_workflow.py --host {args.host} --dataset_id {args.dataset_id} --key {args.key}",
        # Path to the local directory that contains the script.py file
        runtime_env={"working_dir": "./", "conda": args.env}
    )
    print(job_id)


    def wait_until_status(job_id, status_to_wait_for, timeout_seconds=10000):
        start = time.time()
        while time.time() - start <= timeout_seconds:
            status = client.get_job_status(job_id)
            print(f"status: {status}")
            if status in status_to_wait_for:
                break
            time.sleep(1)


    wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
    logs = client.get_job_logs(job_id)
    print(logs)

    # Finish
    logging.warning("Done")
