# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
#
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is
# regenerated.
# --------------------------------------------------------------------------

from typing import Any, Optional, TYPE_CHECKING

from azure.mgmt.core import AsyncARMPipelineClient
from azure.profiles import KnownProfiles, ProfileDefinition
from azure.profiles.multiapiclient import MultiApiClientMixin

from .._serialization import Deserializer, Serializer
from ._configuration import TemplateSpecsClientConfiguration

if TYPE_CHECKING:
    # pylint: disable=unused-import,ungrouped-imports
    from azure.core.credentials_async import AsyncTokenCredential

class _SDKClient(object):
    def __init__(self, *args, **kwargs):
        """This is a fake class to support current implemetation of MultiApiClientMixin."
        Will be removed in final version of multiapi azure-core based client
        """
        pass

class TemplateSpecsClient(MultiApiClientMixin, _SDKClient):
    """The APIs listed in this specification can be used to manage Template Spec resources through the Azure Resource Manager.

    This ready contains multiple API versions, to help you deal with all of the Azure clouds
    (Azure Stack, Azure Government, Azure China, etc.).
    By default, it uses the latest API version available on public Azure.
    For production, you should stick to a particular api-version and/or profile.
    The profile sets a mapping between an operation group and its API version.
    The api-version parameter sets the default API version if the operation
    group is not described in the profile.

    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param subscription_id: Subscription Id which forms part of the URI for every service call. Required.
    :type subscription_id: str
    :param api_version: API version to use if no profile is provided, or if missing in profile.
    :type api_version: str
    :param base_url: Service URL
    :type base_url: str
    :param profile: A profile definition, from KnownProfiles to dict.
    :type profile: azure.profiles.KnownProfiles
    """

    DEFAULT_API_VERSION = '2022-02-01'
    _PROFILE_TAG = "azure.mgmt.resource.templatespecs.TemplateSpecsClient"
    LATEST_PROFILE = ProfileDefinition({
        _PROFILE_TAG: {
            None: DEFAULT_API_VERSION,
        }},
        _PROFILE_TAG + " latest"
    )

    def __init__(
        self,
        credential: "AsyncTokenCredential",
        subscription_id: str,
        api_version: Optional[str] = None,
        base_url: str = "https://management.azure.com",
        profile: KnownProfiles = KnownProfiles.default,
        **kwargs  # type: Any
    ) -> None:
        self._config = TemplateSpecsClientConfiguration(credential, subscription_id, **kwargs)
        self._client = AsyncARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        super(TemplateSpecsClient, self).__init__(
            api_version=api_version,
            profile=profile
        )

    @classmethod
    def _models_dict(cls, api_version):
        return {k: v for k, v in cls.models(api_version).__dict__.items() if isinstance(v, type)}

    @classmethod
    def models(cls, api_version=DEFAULT_API_VERSION):
        """Module depends on the API version:

           * 2019-06-01-preview: :mod:`v2019_06_01_preview.models<azure.mgmt.resource.templatespecs.v2019_06_01_preview.models>`
           * 2021-03-01-preview: :mod:`v2021_03_01_preview.models<azure.mgmt.resource.templatespecs.v2021_03_01_preview.models>`
           * 2021-05-01: :mod:`v2021_05_01.models<azure.mgmt.resource.templatespecs.v2021_05_01.models>`
           * 2022-02-01: :mod:`v2022_02_01.models<azure.mgmt.resource.templatespecs.v2022_02_01.models>`
        """
        if api_version == '2019-06-01-preview':
            from ..v2019_06_01_preview import models
            return models
        elif api_version == '2021-03-01-preview':
            from ..v2021_03_01_preview import models
            return models
        elif api_version == '2021-05-01':
            from ..v2021_05_01 import models
            return models
        elif api_version == '2022-02-01':
            from ..v2022_02_01 import models
            return models
        raise ValueError("API version {} is not available".format(api_version))

    @property
    def template_spec_versions(self):
        """Instance depends on the API version:

           * 2019-06-01-preview: :class:`TemplateSpecVersionsOperations<azure.mgmt.resource.templatespecs.v2019_06_01_preview.aio.operations.TemplateSpecVersionsOperations>`
           * 2021-03-01-preview: :class:`TemplateSpecVersionsOperations<azure.mgmt.resource.templatespecs.v2021_03_01_preview.aio.operations.TemplateSpecVersionsOperations>`
           * 2021-05-01: :class:`TemplateSpecVersionsOperations<azure.mgmt.resource.templatespecs.v2021_05_01.aio.operations.TemplateSpecVersionsOperations>`
           * 2022-02-01: :class:`TemplateSpecVersionsOperations<azure.mgmt.resource.templatespecs.v2022_02_01.aio.operations.TemplateSpecVersionsOperations>`
        """
        api_version = self._get_api_version('template_spec_versions')
        if api_version == '2019-06-01-preview':
            from ..v2019_06_01_preview.aio.operations import TemplateSpecVersionsOperations as OperationClass
        elif api_version == '2021-03-01-preview':
            from ..v2021_03_01_preview.aio.operations import TemplateSpecVersionsOperations as OperationClass
        elif api_version == '2021-05-01':
            from ..v2021_05_01.aio.operations import TemplateSpecVersionsOperations as OperationClass
        elif api_version == '2022-02-01':
            from ..v2022_02_01.aio.operations import TemplateSpecVersionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'template_spec_versions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def template_specs(self):
        """Instance depends on the API version:

           * 2019-06-01-preview: :class:`TemplateSpecsOperations<azure.mgmt.resource.templatespecs.v2019_06_01_preview.aio.operations.TemplateSpecsOperations>`
           * 2021-03-01-preview: :class:`TemplateSpecsOperations<azure.mgmt.resource.templatespecs.v2021_03_01_preview.aio.operations.TemplateSpecsOperations>`
           * 2021-05-01: :class:`TemplateSpecsOperations<azure.mgmt.resource.templatespecs.v2021_05_01.aio.operations.TemplateSpecsOperations>`
           * 2022-02-01: :class:`TemplateSpecsOperations<azure.mgmt.resource.templatespecs.v2022_02_01.aio.operations.TemplateSpecsOperations>`
        """
        api_version = self._get_api_version('template_specs')
        if api_version == '2019-06-01-preview':
            from ..v2019_06_01_preview.aio.operations import TemplateSpecsOperations as OperationClass
        elif api_version == '2021-03-01-preview':
            from ..v2021_03_01_preview.aio.operations import TemplateSpecsOperations as OperationClass
        elif api_version == '2021-05-01':
            from ..v2021_05_01.aio.operations import TemplateSpecsOperations as OperationClass
        elif api_version == '2022-02-01':
            from ..v2022_02_01.aio.operations import TemplateSpecsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'template_specs'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    async def close(self):
        await self._client.close()
    async def __aenter__(self):
        await self._client.__aenter__()
        return self
    async def __aexit__(self, *exc_details):
        await self._client.__aexit__(*exc_details)
