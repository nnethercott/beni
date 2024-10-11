from enum import Enum
from google.cloud import aiplatform
import math
import logging
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random,
    after_log,
    retry_if_exception,
)
from time import gmtime, strftime
from typing import Optional, Tuple
import uuid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Store your full project ID in a variable in the format the API needs.
PROJECT_ID = "deepo-research"
BUCKET_URI = "gs://vlm-artifacts"

aiplatform.init(
    project=PROJECT_ID,
    staging_bucket=BUCKET_URI,
)


class GPUCount(Enum):
    """
    # Enum to pass to the launch function for argument 'gpu_count'.
    # Describes the number of GPU reserved on the training server: 1, 4 or 8
    """

    X1 = 1
    X2 = 2
    X4 = 4
    X8 = 8


class GPUType(Enum):
    """
    Enum to pass to the launch function for argument 'gpu_type'.
    Describes the type of GPU reserved on the training server:
    - None (CPU)
    - K80
    - P100
    - V100
    """

    NONE = ""
    K80 = "NVIDIA_TESLA_K80"
    P4 = "NVIDIA_TESLA_P4"
    T4 = "NVIDIA_TESLA_T4"
    L4 = "NVIDIA_L4"
    P100 = "NVIDIA_TESLA_P100"
    V100 = "NVIDIA_TESLA_V100"


class Region(Enum):
    """
    Enum to choose the region in which to deploy.
    Google has some weird GPU deployment plan: all GPUs
    won't be accessible in all regions.
    """

    EUROPE = "europe"


gpu_to_az = {
    Region.EUROPE: {
        GPUType.NONE: "west1",
        GPUType.T4: "west1",
        GPUType.L4: "west4",
        GPUType.P100: "west1",
        GPUType.V100: "west4",
    }
}


gpu_to_machine_type = {
    GPUType.NONE: ("n1-highcpu", 16),
    GPUType.K80: ("n1-standard", 4),
    GPUType.P4: ("n1-standard", 2),
    GPUType.T4: ("n1-standard", 4),
    GPUType.P100: ("n1-standard", 4),
    GPUType.V100: ("n1-standard", 4),
    GPUType.L4: ("g2-standard", 4),  # new
}

default_retry_parameters = {
    "reraise": True,
    # "retry": retry_if_exception(retry_if_io_error),
    "stop": stop_after_attempt(5),
    # Quota for cloudML are computed every 60 seconds
    # So you need to wait for the next minute if you reach quota
    "wait": wait_exponential(min=60, multiplier=40) + wait_random(0, 10),
    "after": after_log(logger, logging.DEBUG),
}


# -----------------------------------------------------------------------------#


def create_unique_identifier():
    return (
        strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        + "_"
        + str(uuid.uuid4()).replace("-", "")
    )


def create_job_dir_and_id(prefix=None):
    """
    Args:
        prefix: If not None, prepends the given prefix to the job ID. Might be useful for CloudML
            which enforce job_ids starting with a letter.
    """
    job_id = create_unique_identifier()
    if prefix is not None:
        job_id = prefix + job_id
    job_dir = os.path.join("gs://dp-thoth/jobs", strftime("%Y", gmtime()), job_id)
    return job_dir, job_id

    # -----------------------------------------------------------------------------#


# @retry(**default_retry_parameters)
def create_training_job(job_name, training_inputs, labels, region):
    """
    API call to create a Google AI Platform training job.

    Args:
        job_id (str): Job ID of the AI Platform job as given during job creation
        training_inputs (dict): API Rest Resource:TrainingInput https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
        labels (dict): optional labels to organise jobs. Each label is a key-value pair, where both the key and the value are arbitrary strings that you supply.

    Returns:
        response (dict): API Rest Resource:Job as defined at https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job

    """
    # location = "us-central1"
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    job_spec = {"display_name": job_name, "job_spec": training_inputs, "labels": labels}

    parent = f"projects/{PROJECT_ID}/locations/{region}"
    response = client.create_custom_job(
        parent=parent,
        custom_job=job_spec,
    )
    print("response:", response)


def define_training_inputs(
    docker_image: Optional[str],
    gpu_count: GPUCount,
    gpu_type: GPUType,
    machine_type: Optional[str],
    cloudml_env: dict[str, str],
) -> dict[str, str]:
    """
    Create the dictionary to define the API Rest Resource:TrainingInput https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput

    Args:
        docker_image: Label Thoth docker image to use. Might be None in which case we build the local package and use that image.
        model_key: Model key for the job to be launched, example : image_classification.pretraining_natural_rgb.softmax.efficientnet_b0
        benchmark_type: Benchmark type to be logged in airtable for easier filtering
        benchmark_campaign: Benchmark campaign name to regroup all related trainings
        app_id: APP ID for deepomatic client
        api_key: API key for deepomatic client
        gpu_count: number of GPU to reserve on the training server
        gpu_type: type of GPU to reserve on the training server
        machine_type: type of GCE machine to use
        region: region in which to deploy
        vulcan_network_name: If not None, deploy the model to Vulcan with that name.
        evaluate_every_epoch: Evaluate the model at each epoch end and select the best epoch for the final model
        evaluate_final_model: Evaluate the model selected at the end of the training
        experiment_file: file name to store the `deepomatic.oef.Experiment` object in
        cloudml_env: environment description to use on the AI Platform
        use_vulcan_staging: use Vulcan staging?

    Returns:
        training_inputs: API Rest Resource:TrainingInput https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
    """
    # masterType
    default_machine_type, default_machine_size = gpu_to_machine_type[gpu_type]
    if machine_type is None:
        gpu_count_factor = 1 + round(math.log2(gpu_count.value))

        machine_type = "{}-{}".format(
            default_machine_type, default_machine_size * gpu_count_factor
        )

    # with open(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), "rb") as f:
    #     data = f.read()
    # b64auth = base64.b64encode(data).decode("utf-8")

    # TODO: add config yamls here
    # args = [
    #     "--experiment-file",
    #     experiment_file,
    #     "--benchmark-type",
    #     benchmark_type,
    #     "--benchmark-campaign",
    #     benchmark_campaign,
    #     "--app-id",
    #     app_id,
    #     "--api-key",
    #     api_key,
    #     "--env_file",
    #     "GOOGLE_APPLICATION_CREDENTIALS={}".format(b64auth),
    # ]
    # args = ["--runtime", "nvidia"]
    args = []

    training_inputs = {
        "worker_pool_specs": [
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": docker_image,
                    "command": [],
                    "args": args,
                },
            }
        ]
    }

    if gpu_type != GPUType.NONE:
        training_inputs["worker_pool_specs"][0]["machine_spec"].update(
            {
                # "accelerator_type": gpu_type.value,
                "accelerator_type": gpu_type.value,
                "accelerator_count": gpu_count.value,
            }
        )

    return training_inputs


def launch(
    docker_image="europe-west1-docker.pkg.dev/deepo-research/research/vlm_training_zoo:latest",
    gpu_count=GPUCount.X1,
    gpu_type=GPUType.NONE,
    machine_type=None,
    region=Region.EUROPE,
    cloudml_env={},
    prefix="job",
    labels={"billing": "research"},
):
    """
    Create the dictionary to define the API Rest Resource:TrainingInput https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput

    Args:
        xp: OEF Experiment to launch
        docker_image (str): Label of Thoth docker image to use. The image must be in
            the `eu.gcr.io/deepomatic-160015/` registry
            (e.g. `eu.gcr.io/deepomatic-160015/thoth-job:123`).
            If None, will use the tensorflow AI Platform environments and
            try to install packages on boot. This is less stable as AI Platform
            environment might differ from our built docker image.

        model_key (str): Model key for the job to be launched, example : image_classification.pretraining_natural_rgb.softmax.efficientnet_b0
        benchmark_type (str): Benchmark type to be logged in airtable for easier filtering
        benchmark_campaign (str): Benchmark campaign name to regroup all related trainings
        app_id (str): APP ID for deepomatic client
        api_key (str): API key for deepomatic client
        gpu_count (enum): number of GPU to reserve on the training server
        gpu_type (enum): type of GPU to reserve on the training server
        machine_type (str): type of GCE machine to use
        region (enum): region in which to deploy

        vulcan_network_name (str): If not None, deploy the model to Vulcan with that name.
        evaluate_every_epoch: Evaluate the model at each epoch end and select the best epoch for the final model
        evaluate_final_model (bool): Evaluate the model selected at the end of the training
        experiment_file (str): file name to store the `deepomatic.oef.Experiment` object in
        cloudml_env (dict): environment description to use on the AI Platform
        prefix (str): prefix to add to the Google AI Platform Job name
        use_vulcan_staging (bool): use Vulcan staging?
        labels (dict): optional labels to organise jobs. Each label is a key-value pair, where both the key and the value are arbitrary strings that you supply.

    Returns:
        training_inputs (dict): API Rest Resource:TrainingInput https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#TrainingInput
    """

    training_inputs = define_training_inputs(
        docker_image,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        machine_type=machine_type,
        cloudml_env=cloudml_env,
    )

    # prefix += "_"
    # remote_dir, job_id = create_job_dir_and_id(prefix=prefix)
    # logger.info(f"Launching cloudml job in {remote_dir}")

    job_name = "beni-" + str(uuid.uuid4())
    create_training_job(job_name, training_inputs, labels, region)

    # return StatusReader(remote_dir, wait_start_fn=wait_job_start_fn)
    return


if __name__ == "__main__":
    gpu_type = GPUType.L4
    launch(
        gpu_type=gpu_type,
        gpu_count=GPUCount.X1,
        region="europe-west4",
    )
