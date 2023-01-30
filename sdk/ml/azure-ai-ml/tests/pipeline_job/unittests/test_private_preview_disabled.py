from pathlib import Path
import pytest
import yaml
from jsonschema.validators import validate
from marshmallow import Schema

from azure.ai.ml.entities import PipelineJob, CommandComponent
from test_utilities.json_schema import PatchedJSONSchema

from .._util import _PIPELINE_JOB_TIMEOUT_SECOND


# schema of nodes will be reloaded with private preview features disabled in unregister_internal_components
@pytest.mark.usefixtures("disable_internal_components")
@pytest.mark.timeout(_PIPELINE_JOB_TIMEOUT_SECOND)
@pytest.mark.unittest
@pytest.mark.pipeline_test
class TestPrivatePreviewDisabled:
    @pytest.mark.parametrize(
        "source_file,target_schema",
        [
            pytest.param(
                "command_component.yaml",
                CommandComponent._create_schema_for_validation(context={"base_path": "./"}),
                id="command_component",
            ),
            pytest.param(
                "pipeline_job.yaml",
                PipelineJob._create_schema_for_validation(context={"base_path": "./"}),
                id="pipeline_job",
            ),
        ],
    )
    def test_public_json_schema(self, source_file: str, target_schema: Schema):
        # public json schema is the json schema to be saved in
        # https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
        target_schema = PatchedJSONSchema().dump(target_schema)

        with open(Path("./tests/test_configs/json_schema_validation").joinpath(source_file), "r") as f:
            yaml_data = yaml.safe_load(f.read())

        validate(yaml_data, target_schema)
