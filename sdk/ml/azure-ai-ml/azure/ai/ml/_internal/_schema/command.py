# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from marshmallow import fields

from azure.ai.ml._internal._schema.node import InternalBaseNodeSchema, NodeType
from azure.ai.ml._schema import (
    NestedField,
    StringTransformedEnum,
    UnionField,
    RegistryStr,
    AnonymousEnvironmentSchema,
    ArmVersionedStr
)
from azure.ai.ml._schema.job import ParameterizedParallelSchema, ParameterizedCommandSchema
from azure.ai.ml._schema.job.job_limits import CommandJobLimitsSchema
from azure.ai.ml.constants._common import AzureMLResourceType


class CommandSchema(InternalBaseNodeSchema, ParameterizedCommandSchema):
    class Meta:
        exclude = ["code", "distribution"]  # internal command doesn't have code & distribution

    environment = UnionField(
        [
            RegistryStr(azureml_type=AzureMLResourceType.ENVIRONMENT),
            NestedField(AnonymousEnvironmentSchema),
            ArmVersionedStr(azureml_type=AzureMLResourceType.ENVIRONMENT, allow_default_version=True),
        ],
    )
    type = StringTransformedEnum(allowed_values=[NodeType.COMMAND], casing_transform=lambda x: x)
    limits = NestedField(CommandJobLimitsSchema)


class DistributedSchema(CommandSchema):
    class Meta:
        exclude = ["code"]  # need to enable distribution comparing to CommandSchema
    type = StringTransformedEnum(allowed_values=[NodeType.DISTRIBUTED], casing_transform=lambda x: x)


class ParallelSchema(InternalBaseNodeSchema, ParameterizedParallelSchema):
    class Meta:
        # partition_keys can still be used with unknown warning, but need to do dump before setting
        exclude = ["task", "input_data", "mini_batch_error_threshold", "partition_keys"]

    type = StringTransformedEnum(allowed_values=[NodeType.PARALLEL], casing_transform=lambda x: x)
    compute = fields.Str()
    environment = fields.Str()
    limits = NestedField(CommandJobLimitsSchema)
