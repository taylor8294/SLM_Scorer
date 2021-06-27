# -*- coding: utf-8 -*-
"""Utility functions."""

from argparse import ArgumentTypeError, FileType

import numpy as np

class FileTypeWithExtensionCheck(FileType):
    def __init__(self, mode='r', valid_extensions=None, **kwargs):
        super().__init__(mode, **kwargs)
        self.valid_extensions = valid_extensions

    def __call__(self, string):
        if self.valid_extensions:
            if not string.lower().endswith(self.valid_extensions):
                print(self)
                raise ArgumentTypeError('File extension must be {}{}'.format(
                    'one of ' if type(self.valid_extensions) is tuple else '',
                    self.valid_extensions
                ))
        return super().__call__(string)


def PolyArea(x, y):
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
