# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._operations import Operations
from ._operations import AvailabilitySetsOperations
from ._operations import ProximityPlacementGroupsOperations
from ._operations import VirtualMachineExtensionImagesOperations
from ._operations import VirtualMachineExtensionsOperations
from ._operations import VirtualMachineImagesOperations
from ._operations import UsageOperations
from ._operations import VirtualMachinesOperations
from ._operations import VirtualMachineSizesOperations
from ._operations import ImagesOperations
from ._operations import VirtualMachineScaleSetsOperations
from ._operations import VirtualMachineScaleSetExtensionsOperations
from ._operations import VirtualMachineScaleSetRollingUpgradesOperations
from ._operations import VirtualMachineScaleSetVMsOperations
from ._operations import LogAnalyticsOperations
from ._operations import VirtualMachineRunCommandsOperations
from ._operations import GalleriesOperations
from ._operations import GalleryImagesOperations
from ._operations import GalleryImageVersionsOperations
from ._operations import DisksOperations
from ._operations import SnapshotsOperations

from ._patch import __all__ as _patch_all
from ._patch import *  # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "Operations",
    "AvailabilitySetsOperations",
    "ProximityPlacementGroupsOperations",
    "VirtualMachineExtensionImagesOperations",
    "VirtualMachineExtensionsOperations",
    "VirtualMachineImagesOperations",
    "UsageOperations",
    "VirtualMachinesOperations",
    "VirtualMachineSizesOperations",
    "ImagesOperations",
    "VirtualMachineScaleSetsOperations",
    "VirtualMachineScaleSetExtensionsOperations",
    "VirtualMachineScaleSetRollingUpgradesOperations",
    "VirtualMachineScaleSetVMsOperations",
    "LogAnalyticsOperations",
    "VirtualMachineRunCommandsOperations",
    "GalleriesOperations",
    "GalleryImagesOperations",
    "GalleryImageVersionsOperations",
    "DisksOperations",
    "SnapshotsOperations",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
