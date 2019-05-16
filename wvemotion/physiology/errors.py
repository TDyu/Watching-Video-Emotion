#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Errors for phsiology.

All errors inherited Exception defined for phsiology.
"""


class SourceTypeError(Exception):
    """Constructor source type is invalid.
    """

    def __init__(self, message):
        super(SourceTypeError, self).__init__(message)
        self.message = message


class SourceDimensionError(Exception):
    """Source data dimension is invalid.
    """

    def __init__(self, message):
        super(SourceDimensionError, self).__init__(message)
        self.message = message
