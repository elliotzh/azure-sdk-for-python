# coding=utf-8
# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for license information.
# Code generated by Microsoft (R) AutoRest Code Generator.
# Changes may cause incorrect behavior and will be lost if the code is regenerated.
# --------------------------------------------------------------------------

from ._models import AnswerSpan
from ._models import AnswersFromTextOptions
from ._models import AnswersFromTextResult
from ._models import AnswersOptions
from ._models import AnswersResult
from ._models import Error
from ._models import ErrorResponse
from ._models import InnerErrorModel
from ._models import KnowledgeBaseAnswer
from ._models import KnowledgeBaseAnswerContext
from ._models import KnowledgeBaseAnswerDialog
from ._models import KnowledgeBaseAnswerPrompt
from ._models import MetadataFilter
from ._models import QueryFilters
from ._models import ShortAnswerOptions
from ._models import TextAnswer
from ._models import TextDocument

from ._enums import ErrorCode
from ._enums import InnerErrorCode
from ._patch import __all__ as _patch_all
from ._patch import *  # type: ignore # pylint: disable=unused-wildcard-import
from ._patch import patch_sdk as _patch_sdk

__all__ = [
    "AnswerSpan",
    "AnswersFromTextOptions",
    "AnswersFromTextResult",
    "AnswersOptions",
    "AnswersResult",
    "Error",
    "ErrorResponse",
    "InnerErrorModel",
    "KnowledgeBaseAnswer",
    "KnowledgeBaseAnswerContext",
    "KnowledgeBaseAnswerDialog",
    "KnowledgeBaseAnswerPrompt",
    "MetadataFilter",
    "QueryFilters",
    "ShortAnswerOptions",
    "TextAnswer",
    "TextDocument",
    "ErrorCode",
    "InnerErrorCode",
]
__all__.extend([p for p in _patch_all if p not in __all__])
_patch_sdk()
