# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import hashlib
import logging
import os
from collections import defaultdict
from typing import Union, Tuple, List

from azure.ai.ml.constants._common import AzureMLResourceType
from azure.ai.ml.entities import Component
from azure.ai.ml.entities._builders import BaseNode


logger = logging.getLogger(__name__)


class CachedNodeResolver(object):
    """Class to resolve component in nodes with cached component resolution results.

    This class is thread-safe if:
    1) self._resolve_nodes and self._register_node_to_resolve are not called concurrently in the same thread
    2) self._resolve_component is only called concurrently on independent components
        a) we have used a component hash to deduplicate components to resolve;
        b) nodes are registered & resolved layer by layer, so all child components are already resolved
          when we register a pipeline node;
        c) potential shared dependencies like compute and data have been resolved before calling self._resolve_nodes.
    """
    _ANONYMOUS_HASH_PREFIX = "anonymous_component:"
    _YAML_SOURCE_PREFIX = "yaml_source:"
    _ORIGIN_PREFIX = "origin:"

    def __init__(self, resolver):
        self._resolver = resolver
        self._component_resolution_cache = {}
        self._components_to_resolve: List[Tuple[str, Component]] = []
        self._nodes_to_apply = defaultdict(list)

    def _resolve_component(self, component: Union[str, Component]) -> str:
        """Resolve a component with self._resolver."""
        return self._resolver(component, azureml_type=AzureMLResourceType.COMPONENT)

    @classmethod
    def _get_component_hash(cls, component: Union[Component, str]):
        """Get a hash for a component."""
        if isinstance(component, str):
            # component can be arm string like "train_model:1"
            return cls._ORIGIN_PREFIX + component

        # For components with code, its code will be an absolute path before uploaded to blob,
        # so we can use a mixture of its anonymous hash and its source path as its hash, in case
        # there are 2 components with same code but different ignore files
        # Here we can check if the component has a source path instead of check if it has code, as
        # there is no harm to add a source path to the hash even if the component doesn't have code
        # Note that here we assume that the content of code folder won't change during the submission
        if component._source_path:  # pylint: disable=protected-access
            object_hash = hashlib.sha224()
            object_hash.update(component._get_anonymous_hash().encode("utf-8"))  # pylint: disable=protected-access
            object_hash.update(component._source_path.encode("utf-8"))  # pylint: disable=protected-access
            return cls._YAML_SOURCE_PREFIX + object_hash.hexdigest()
        # For components without code, like pipeline component, their dependencies have already
        # been resolved before calling this function, so we can use their anonymous hash directly
        return cls._ANONYMOUS_HASH_PREFIX + component._get_anonymous_hash()  # pylint: disable=protected-access

    @classmethod
    def _get_component_registration_max_workers(cls):
        # Before Python 3.8, the default max_worker is the number of processors multiplied by 5.
        # It may send a large number of the uploading snapshot requests that will occur remote refuses requests.
        # In order to avoid retrying the upload requests, max_worker will use the default value in Python 3.8,
        # min(32, os.cpu_count + 4).
        max_workers_env_var = 'AML_COMPONENT_REGISTRATION_MAX_WORKER'
        default_max_workers = min(32, (os.cpu_count() or 1) + 4)
        try:
            max_workers = int(os.environ.get(max_workers_env_var, default_max_workers))
        except ValueError:
            logger.info(
                "Environment variable %s with value %s set but failed to parse. "
                "Use the default max_worker %s as registration thread pool max_worker."
                "Please reset the value to an integer.",
                max_workers_env_var,
                os.environ.get(max_workers_env_var),
                default_max_workers
            )
            max_workers = default_max_workers
        return max_workers

    def register_node_to_resolve(self, node: BaseNode):
        """Register a node with its component to resolve.
        """
        component = node._component  # pylint: disable=protected-access
        # 1 possible optimization is to save an id(component) -> component hash map here to
        # avoid duplicate hash calculation. The risk is that we haven't implicitly forbidden component
        # modification after it's been used in a node, and we can't guarantee that the hash is still
        # valid after modification.
        component_hash = self._get_component_hash(component)
        if component_hash not in self._component_resolution_cache:
            self._components_to_resolve.append((component_hash, component))
            # set corresponding resolution cache to None to avoid duplicate resolution
            self._component_resolution_cache[component_hash] = None

        self._nodes_to_apply[component_hash].append(node)

    def _resolve_components_without_in_memory_cache(self):
        """Resolve all components to resolve and save the results in cache.
        """
        # TODO: Shouldn't reach this function when trying to resolve a subgraph
        # TODO: apply local cache controlled by an environment variable here
        # TODO: do concurrent resolution controlled by an environment variable here
        # given deduplication has already been done, we can safely assume that there is no
        # conflict in concurrent local cache access
        # pool = multiprocessing.Pool(self._get_component_registration_max_workers())
        # results = pool.map(self._resolve_component, [component for _, component in self._components_to_resolve])
        # for (component_hash, component), result in zip(self._components_to_resolve, results):
        #     self._component_resolution_cache[component_hash] = result

        for component_hash, component in self._components_to_resolve:
            self._component_resolution_cache[component_hash] = self._resolve_component(component)

        self._components_to_resolve.clear()

    def resolve_nodes(self):
        """Resolve all dependent components with resolver and set resolved component arm id back to newly
        registered nodes. Registered nodes will be cleared after resolution.
        """
        if self._components_to_resolve:
            self._resolve_components_without_in_memory_cache()

        for component_hash, nodes in self._nodes_to_apply.items():
            for node in nodes:
                node._component = self._component_resolution_cache[component_hash]  # pylint: disable=protected-access
        self._nodes_to_apply.clear()
