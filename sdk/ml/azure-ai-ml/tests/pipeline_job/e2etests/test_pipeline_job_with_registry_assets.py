import os.path
from pathlib import Path
from typing import Any, Callable, Dict

from devtools_testutils import AzureRecordedTestCase
from devtools_testutils import is_live
import pydash
import pytest
from test_utilities.utils import (
    assert_job_cancel,
    sleep_if_live,
    wait_until_done,
    _PYTEST_TIMEOUT_METHOD,
)

from azure.ai.ml import Input, MLClient, load_component, load_data, load_job
from azure.ai.ml._utils._arm_id_utils import AMLVersionedArmId
from azure.ai.ml._utils.utils import load_yaml
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.constants._job.pipeline import PipelineConstants
from azure.ai.ml.entities import Component, Job, PipelineJob
from azure.ai.ml.entities._builders import Command, Pipeline
from azure.ai.ml.entities._builders.parallel import Parallel
from azure.ai.ml.entities._builders.spark import Spark
from azure.ai.ml.exceptions import JobException
from azure.core.exceptions import HttpResponseError

from .._util import (
    _PIPELINE_JOB_LONG_RUNNING_TIMEOUT_SECOND,
    _PIPELINE_JOB_TIMEOUT_SECOND,
    DATABINDING_EXPRESSION_TEST_CASES,
    DATABINDING_EXPRESSION_TEST_CASE_ENUMERATE,
)


def assert_job_input_output_types(job: PipelineJob):
    from azure.ai.ml.entities._job.pipeline._io import NodeInput, NodeOutput, PipelineInput, PipelineOutput

    for _, input in job.inputs.items():
        assert isinstance(input, PipelineInput)
    for _, output in job.outputs.items():
        assert isinstance(output, PipelineOutput)
    for _, component in job.jobs.items():
        for _, input in component.inputs.items():
            assert isinstance(input, NodeInput)
        for _, output in component.outputs.items():
            assert isinstance(output, NodeOutput)


@pytest.mark.usefixtures(
    "recorded_test",
    "mock_code_hash",
    "enable_pipeline_private_preview_features",
    "mock_asset_name",
    "mock_component_hash",
    "enable_environment_id_arm_expansion",
)
@pytest.mark.timeout(timeout=_PIPELINE_JOB_TIMEOUT_SECOND, method=_PYTEST_TIMEOUT_METHOD)
@pytest.mark.e2etest
@pytest.mark.pipeline_test
class TestPipelineJobWithRegistryAssets(AzureRecordedTestCase):
    @pytest.mark.skip(
        reason="request body still exits when re-record and will raise error "
        "'Unable to find a record for the request' in playback mode"
    )
    def test_pipeline_job_create_with_registry_model_as_input(
        self,
        client: MLClient,
        registry_client: MLClient,
        randstr: Callable[[str], str],
    ) -> None:
        params_override = [{"name": randstr("name")}]
        pipeline_job = load_job(
            source="./tests/test_configs/pipeline_jobs/job_with_registry_model_as_input/pipeline.yml",
            params_override=params_override,
        )
        job = client.jobs.create_or_update(pipeline_job)
        assert job.name == params_override[0]["name"]

    def test_pipeline_job_create_with_registries(
        self,
        client: MLClient,
        randstr: Callable[[str], str],
    ) -> None:
        params_override = [{"name": randstr("name")}]
        pipeline_job = load_job(
            source="./tests/test_configs/pipeline_jobs/hello_pipeline_job_with_registries.yml",
            params_override=params_override,
        )
        values_to_check = [
            (
                "environment_on_registry",
                "environment",
                "azureml://registries/testFeed/environments/sklearn-10-ubuntu2004-py38-cpu/versions/19.dev6"
            ),
            (
                "component_on_registry",
                "component",
                "azureml://registries/testFeed/components/sample_command_component_basic/versions/1"
            ),
            (
                "component_on_registry_via_label",
                "component",
                "azureml://registries/testFeed/components/sample_command_component_basic/labels/latest"
            ),
            (
                "component_on_registry_via_label_2",
                "component",
                "azureml://registries/testFeed/components/sample_command_component_basic@latest"
            ),
        ]

        for node_name, key, expected_value in values_to_check:
            assert getattr(pipeline_job.jobs.get(node_name), key) == expected_value
        created_job = client.jobs.create_or_update(pipeline_job)
        assert created_job.name == params_override[0]["name"]
        for node_name, key, expected_value in values_to_check:
            assert getattr(created_job.jobs.get(node_name), key) == expected_value

