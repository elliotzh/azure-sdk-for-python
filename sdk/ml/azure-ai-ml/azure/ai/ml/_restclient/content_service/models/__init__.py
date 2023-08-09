# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

try:
    from ._models_py3 import (
        CreateSnapshot,
        CreateSnapshotFromUri,
        DirTreeNode,
        ErrorAdditionalInfo,
        ErrorResponse,
        FileNameList,
        FileNode,
        FileNodeList,
        FlatDirTreeNode,
        FlatDirTreeNodeList,
        GitCredential,
        GitRevision,
        InnerErrorResponse,
        MerkleDiffEntry,
        RootError,
        Snapshot,
        SnapshotConsumptionRequest,
        SnapshotCredentials,
        SnapshotList,
        SnapshotMetadata,
        User,
        ValueTuple2,
    )
except (SyntaxError, ImportError):
    from ._models import CreateSnapshot  # type: ignore
    from ._models import CreateSnapshotFromUri  # type: ignore
    from ._models import DirTreeNode  # type: ignore
    from ._models import ErrorAdditionalInfo  # type: ignore
    from ._models import ErrorResponse  # type: ignore
    from ._models import FileNameList  # type: ignore
    from ._models import FileNode  # type: ignore
    from ._models import FileNodeList  # type: ignore
    from ._models import FlatDirTreeNode  # type: ignore
    from ._models import FlatDirTreeNodeList  # type: ignore
    from ._models import GitCredential  # type: ignore
    from ._models import GitRevision  # type: ignore
    from ._models import InnerErrorResponse  # type: ignore
    from ._models import MerkleDiffEntry  # type: ignore
    from ._models import RootError  # type: ignore
    from ._models import Snapshot  # type: ignore
    from ._models import SnapshotConsumptionRequest  # type: ignore
    from ._models import SnapshotCredentials  # type: ignore
    from ._models import SnapshotList  # type: ignore
    from ._models import SnapshotMetadata  # type: ignore
    from ._models import User  # type: ignore
    from ._models import ValueTuple2  # type: ignore

from ._azure_machine_learning_workspaces_enums import (
    GitCredentialType,
    OperationType,
    SnapshotProvisioningState,
    SnapshotType,
)

__all__ = [
    'CreateSnapshot',
    'CreateSnapshotFromUri',
    'DirTreeNode',
    'ErrorAdditionalInfo',
    'ErrorResponse',
    'FileNameList',
    'FileNode',
    'FileNodeList',
    'FlatDirTreeNode',
    'FlatDirTreeNodeList',
    'GitCredential',
    'GitRevision',
    'InnerErrorResponse',
    'MerkleDiffEntry',
    'RootError',
    'Snapshot',
    'SnapshotConsumptionRequest',
    'SnapshotCredentials',
    'SnapshotList',
    'SnapshotMetadata',
    'User',
    'ValueTuple2',
    'GitCredentialType',
    'OperationType',
    'SnapshotProvisioningState',
    'SnapshotType',
]
