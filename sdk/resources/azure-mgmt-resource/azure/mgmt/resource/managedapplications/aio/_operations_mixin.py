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
from .._serialization import Serializer, Deserializer
from typing import Any, AsyncIterable

from azure.core.async_paging import AsyncItemPaged

from .. import models as _models


class ApplicationClientOperationsMixin(object):

    def list_operations(
        self,
        **kwargs: Any
    ) -> AsyncIterable["_models.Operation"]:
        """Lists all of the available Microsoft.Solutions REST API operations.

        :keyword callable cls: A custom type or function that will be passed the direct response
        :return: An iterator like instance of either Operation or the result of cls(response)
        :rtype:
         ~azure.core.async_paging.AsyncItemPaged[~azure.mgmt.resource.managedapplications.v2018_06_01.models.Operation]
        :raises ~azure.core.exceptions.HttpResponseError:
        """
        api_version = self._get_api_version('list_operations')
        if api_version == '2018-06-01':
            from ..v2018_06_01.aio.operations import ApplicationClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_operations'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_operations(**kwargs)
