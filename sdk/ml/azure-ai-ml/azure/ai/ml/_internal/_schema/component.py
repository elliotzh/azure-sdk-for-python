# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os.path

import pydash
from marshmallow import EXCLUDE, INCLUDE, fields, post_dump, pre_load

from azure.ai.ml._schema import NestedField, StringTransformedEnum, UnionField
from azure.ai.ml._schema.component.component import ComponentSchema
from azure.ai.ml._schema.core.fields import ArmVersionedStr, CodeField
from azure.ai.ml.constants._common import LABELLED_RESOURCE_NAME, AzureMLResourceType, SOURCE_PATH_CONTEXT_KEY

from .._utils import yaml_safe_load_with_base_resolver
from ..._utils._arm_id_utils import parse_name_label
from .environment import InternalEnvironmentSchema
from .input_output import (
    InternalEnumParameterSchema,
    InternalInputPortSchema,
    InternalOutputPortSchema,
    InternalParameterSchema,
    InternalPrimitiveOutputSchema,
)


class NodeType:
    COMMAND = "CommandComponent"
    DATA_TRANSFER = "DataTransferComponent"
    DISTRIBUTED = "DistributedComponent"
    HDI = "HDInsightComponent"
    PARALLEL = "ParallelComponent"
    SCOPE = "ScopeComponent"
    STARLITE = "StarliteComponent"
    SWEEP = "SweepComponent"
    PIPELINE = "PipelineComponent"
    HEMERA = "HemeraComponent"
    AE365EXEPOOL = "AE365ExePoolComponent"
    IPP = "IntellectualPropertyProtectedComponent"

    @classmethod
    def all_values(cls):
        all_values = []
        for key, value in vars(cls).items():
            if not key.startswith("_") and isinstance(value, str):
                all_values.append(value)
        return all_values


def get_all_valid_dot_keys(left_reversed_parts, validate_func, root, *, cur_node=None, processed_parts=None):
    if len(left_reversed_parts) == 0:
        if validate_func(root, processed_parts):
            return [".".join(processed_parts)]
        return []

    if cur_node is None:
        cur_node = root
    if not isinstance(cur_node, dict):
        return []
    if processed_parts is None:
        processed_parts = []

    key: str = left_reversed_parts.pop()
    result = []
    if key == "*":
        for next_key in cur_node:
            if not isinstance(next_key, str):
                continue
            processed_parts.append(next_key)
            result.extend(get_all_valid_dot_keys(
                left_reversed_parts,
                validate_func,
                root,
                cur_node=cur_node[next_key],
                processed_parts=processed_parts)
            )
            processed_parts.pop()
    elif key in cur_node:
        processed_parts.append(key)
        result = get_all_valid_dot_keys(
            left_reversed_parts,
            validate_func,
            root,
            cur_node=cur_node[key],
            processed_parts=processed_parts
        )
        processed_parts.pop()

    left_reversed_parts.append(key)
    return result


class InternalComponentSchema(ComponentSchema):
    class Meta:
        unknown = INCLUDE

    # override name as 1p components allow . in name, which is not allowed in v2 components
    name = fields.Str()

    # override to allow empty properties
    tags = fields.Dict(keys=fields.Str())

    # override inputs & outputs to support 1P inputs & outputs, may need to do strict validation later
    # no need to check io type match since server will do that
    inputs = fields.Dict(
        keys=fields.Str(),
        values=UnionField(
            [
                NestedField(InternalParameterSchema),
                NestedField(InternalEnumParameterSchema),
                NestedField(InternalInputPortSchema),
            ]
        ),
    )
    # support primitive output for all internal components for now
    outputs = fields.Dict(
        keys=fields.Str(),
        values=UnionField(
            [
                NestedField(InternalPrimitiveOutputSchema, unknown=EXCLUDE),
                NestedField(InternalOutputPortSchema, unknown=EXCLUDE),
            ]
        ),
    )

    # type field is required for registration
    type = StringTransformedEnum(
        allowed_values=NodeType.all_values(),
        casing_transform=lambda x: parse_name_label(x)[0],
        pass_original=True,
    )

    # need to resolve as it can be a local field
    code = CodeField()

    environment = UnionField(
        [
            ArmVersionedStr(azureml_type=AzureMLResourceType.ENVIRONMENT),
            NestedField(InternalEnvironmentSchema),
        ]
    )

    def get_skip_fields(self):  # pylint: disable=no-self-use
        return ["properties"]

    def _serialize(self, obj, *, many: bool = False):
        # pylint: disable=no-member
        if many and obj is not None:
            return super(InternalComponentSchema, self)._serialize(obj, many=many)
        ret = super(InternalComponentSchema, self)._serialize(obj)
        for attr_name in obj.__dict__.keys():
            if (
                not attr_name.startswith("_")
                and attr_name not in self.get_skip_fields()
                and attr_name not in self.dump_fields
            ):
                ret[attr_name] = self.get_attribute(obj, attr_name, None)
        return ret

    # override param_override to ensure that param override happens after reloading the yaml
    @pre_load
    def add_param_overrides(self, data, **kwargs):
        source_path = self.context.pop(SOURCE_PATH_CONTEXT_KEY, None)
        if isinstance(data, dict) and source_path and os.path.isfile(source_path):
            def should_node_overwritten(_root, _parts):
                parts = _parts.copy()
                parts.pop()
                parts.append("type")
                _input_type = pydash.get(_root, parts, None)
                return isinstance(_input_type, str) and _input_type.lower() not in ["boolean"]

            # do override here
            with open(source_path, "r") as f:
                origin_data = yaml_safe_load_with_base_resolver(f)
                for wild_dot_key, condition_func in [
                    ("version", lambda _root, _parts: True),
                    ("inputs.*.default", should_node_overwritten),
                    ("inputs.*.enum", should_node_overwritten),
                ]:
                    for dot_key in get_all_valid_dot_keys(
                        wild_dot_key.split(".")[::-1],
                        condition_func,
                        origin_data,
                    ):
                        pydash.set_(data, dot_key, pydash.get(origin_data, dot_key))
        return super().add_param_overrides(data, **kwargs)

    @post_dump(pass_original=True)
    def simplify_input_output_port(self, data, original, **kwargs):  # pylint:disable=unused-argument, no-self-use
        # remove None in input & output
        for io_ports in [data["inputs"], data["outputs"]]:
            for port_name, port_definition in io_ports.items():
                io_ports[port_name] = dict(filter(lambda item: item[1] is not None, port_definition.items()))

        # hack, to match current serialization match expectation
        for port_name, port_definition in data["inputs"].items():
            if "mode" in port_definition:
                del port_definition["mode"]

        return data

    @post_dump(pass_original=True)
    def add_back_type_label(self, data, original, **kwargs):  # pylint:disable=unused-argument, no-self-use
        type_label = original._type_label  # pylint:disable=protected-access
        if type_label:
            data["type"] = LABELLED_RESOURCE_NAME.format(data["type"], type_label)
        return data
