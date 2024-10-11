from google.cloud import aiplatform
import logging
import uuid
from dotenv import dotenv_values
import yaml
import json

# https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/custom/custom_training_container_and_model_registry.ipynb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "deepo-research"
BUCKET_URI = "gs://another-vlm-artifacts"  # located in europe-west4
LOCATION = "europe-west4"
IMAGE_URI = (
    "europe-west1-docker.pkg.dev/deepo-research/research/vlm_training_zoo:latest"
)


# NOTE: the datasets don't need to be located on the same bucket as the staging_bucket, just public i believe

# TODO: change accelerator_type to T4 for 8:1 fp16 speedup

# def get_args(file: str) -> list[str]:
#     with open(file, "r") as f:
#         args = yaml.safe_load(f.read())

#     # for dealing with json fields
#     def format_value(v):
#         try:
#             json.loads(v)
#             return f"'{v}'"
#         except (json.JSONDecodeError, TypeError):
#             return str(v)

#     args = [f"--{k}={format_value(v)}" for k, v in args.items()]
#     return args


def get_args(file: str) -> list[str]:
    with open(file, "r") as f:
        args = yaml.safe_load(f.read())

    args = [f"--{k}={str(v)}" for k, v in args.items()]
    return args


if __name__ == "__main__":
    pass
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)
    # TODO: change accelerator_type to T4 for 8:1 fp16 speedup
    base_output_dir = BUCKET_URI + "/" + str(uuid.uuid4()).split("-")[0]

    job = aiplatform.CustomContainerTrainingJob(
        display_name="vlm",
        container_uri=IMAGE_URI,
        location=LOCATION,
        staging_bucket=BUCKET_URI,
        project=PROJECT_ID,
        labels={"billing": "research"},
    )

    environment = dict(dotenv_values())
    args = get_args("./train_config.yaml")
    print(environment)
    print(args)

    job.run(
        args=args,
        environment_variables=environment,
        base_output_dir=base_output_dir,
        replica_count=1,
        machine_type="g2-standard-48",
        accelerator_type=aiplatform.gapic.AcceleratorType.NVIDIA_L4.name,
        accelerator_count=4,
        sync=False,
    )
