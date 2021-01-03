# -*- coding: utf-8 -*-
# ====================================== #
# @Author  : Yanbo Han
# @Email   : yanbohan98@gmail.com
# @File    : __init__.py.py
# ALL RIGHTS ARE RESERVED UNLESS STATED.
# ====================================== #

import logging

__all__ = [
    'InfoLogger'
]


class MyLogging(logging.Logger):
    def __init__(self, name, level=logging.INFO, file=None):
        """

        Parameters
        ----------
        name: str
            Name of logger
        level: int
            logging.INFO, logging.DEBUG,...
        file: str
            File name of log.
            If None, log to console.
        """
        super().__init__(name, level)

        # format set of log
        fmt = "[%(asctime)s]-[%(name)s]-[%(levelname)s]-[%(filename)s,line %(lineno)d] : %(message)s"
        formatter = logging.Formatter(fmt)

        # to file
        if file:
            file_handle = logging.FileHandler(file, encoding="utf-8")
            file_handle.setFormatter(formatter)
            self.addHandler(file_handle)
        # to console
        else:
            console_handle = logging.StreamHandler()
            console_handle.setFormatter(formatter)
            self.addHandler(console_handle)


class DefaultInfoLogger(MyLogging):
    """
    Default Logger to file/console with info level.
    """

    def __init__(self, name, file):
        super(DefaultInfoLogger, self).__init__(name=name, level=logging.INFO, file=file)


class InfoLogger(DefaultInfoLogger):
    """
    Default Logger to console with info level.
    """

    def __init__(self, name):
        super(InfoLogger, self).__init__(name=name, file=None)


class DebugLogger(MyLogging):
    """
    Default Logger to console with debug level.
    """

    def __init__(self, name):
        super(DebugLogger, self).__init__(name=name, level=logging.DEBUG, file=None)
