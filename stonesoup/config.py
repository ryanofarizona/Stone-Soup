# -*- coding: utf-8 -*-
"""Provides an ability to generate and load configuration from YAML.

Stone Soup utilises YAML_ for configuration files. The :doc:`stonesoup.base`
feature of components is exploited in order to store the configuration of the
components used for a run.

.. _YAML: http://yaml.org/"""
import datetime
import warnings
from abc import ABC, abstractmethod
from io import StringIO
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from importlib import import_module

import numpy as np
from ruamel.yaml import YAML

from .base import Base


class Configuration:
    pass


class ConfigurationFile(ABC):
    """Base configuration class."""

    @abstractmethod
    def dump(self, data, stream, *args, **kwargs):
        """Dump configuration to a stream."""
        raise NotImplementedError

    def dumps(self, data, *args, **kwargs):
        """Return configuration as a string."""
        stream = StringIO()
        self.dump(data, stream, *args, **kwargs)
        return stream.getvalue()

    @abstractmethod
    def load(self, stream):
        """Load configuration from a stream."""
        raise NotImplementedError


class YAMLConfigurationFile(ConfigurationFile):
    """Class for YAML Configuration file."""
    tag_prefix = '!{}.'.format(__name__.split('.', 1)[0])

    def __init__(self):
        self._yaml = YAML()
        self._yaml.default_flow_style = False

        # Declarative classes
        self._yaml.representer.add_multi_representer(
            Base, self.declarative_to_yaml)
        self._yaml.constructor.add_multi_constructor(
            self.tag_prefix, self.declarative_from_yaml)

        # NumPy
        self._yaml.representer.add_representer(
            np.ndarray, self.ndarray_to_yaml)
        self._yaml.constructor.add_constructor(
            "!numpy.ndarray", self.ndarray_from_yaml)

        # Datetime
        self._yaml.representer.add_representer(
            datetime.timedelta, self.timedelta_to_yaml)
        self._yaml.constructor.add_constructor(
            "!datetime.timedelta", self.timedelta_from_yaml)

        # Path
        self._yaml.representer.add_multi_representer(
            Path, self.path_to_yaml)
        self._yaml.constructor.add_constructor(
            "!pathlib.Path", self.path_from_yaml)

    def dump(self, data, stream, **kwargs):
        return self._yaml.dump(data, stream, **kwargs)

    def load(self, stream):
        return self._yaml.load(stream)

    @classmethod
    def yaml_tag(cls, class_):
        """Return YAML tag for tag for framework configuration file.

        Constructed from module and class name."""
        return "!{}.{}".format(class_.__module__, class_.__qualname__)

    @classmethod
    def declarative_to_yaml(cls, representer, node):
        """Convert declarative class instances to YAML.

        Store as mapping of declared properties, skipping any which are the
        default value."""
        return representer.represent_omap(
            cls.yaml_tag(type(node)),
            OrderedDict((name, getattr(node, name))
                        for name, property_ in type(node).properties.items()
                        if getattr(node, name) is not property_.default))

    @classmethod
    def declarative_from_yaml(cls, constructor, tag_suffix, node):
        """Convert YAML to declarative class instances."""
        class_ = cls._get_class(tag_suffix)
        properties = [
            data
            for data in constructor.construct_yaml_omap(node)][0]
        return class_(**properties)

    @classmethod
    @lru_cache(None)
    def _get_class(cls, tag_suffix):
        tag = cls.tag_prefix + tag_suffix
        classes = [
            subclass
            for subclass in Base.subclasses
            if cls.yaml_tag(subclass) == tag]
        if len(classes) > 1:
            warnings.warn(
                "Multiple possible classes found for YAML tag {!r}".format(
                    tag), UserWarning)
        elif not classes:
            module_name, class_name = tag_suffix.rsplit(".", 1)
            module = import_module("..{}".format(module_name), __name__)
            classes = [getattr(module, class_name, None)]
        return classes[0]

    def ndarray_to_yaml(self, representer, node):
        """Convert numpy.ndarray to YAML."""
        array = [self._yaml.seq(row) for row in node.tolist()]
        [seq.fa.set_flow_style() for seq in array]
        return representer.represent_sequence(
            "!numpy.ndarray", array)

    @staticmethod
    def ndarray_from_yaml(constructor, node):
        """Convert YAML to numpy.ndarray."""
        return np.array(constructor.construct_sequence(node, deep=True))

    @staticmethod
    def timedelta_to_yaml(representer, node):
        """Convert datetime.timedelta to YAML.

        Value is total number of seconds."""
        return representer.represent_scalar(
            "!datetime.timedelta", str(node.total_seconds()))

    @staticmethod
    def timedelta_from_yaml(constructor, node):
        """Convert YAML to datetime.timedelta.

        Value should be total number of seconds."""
        return datetime.timedelta(
            seconds=float(constructor.construct_scalar(node)))

    @staticmethod
    def path_to_yaml(representer, node):
        """Convert path to YAML.

        Value is total number of seconds."""
        return representer.represent_scalar(
            "!pathlib.Path", str(node))

    @staticmethod
    def path_from_yaml(constructor, node):
        """Convert YAML to datetime.timedelta.

        Value should be total number of seconds."""
        return Path(constructor.construct_scalar(node))
