# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper methods to read and process config files."""

import collections
import copy
import operator
import itertools
import glob
import os


def _binary_op(op, reverse=False):

    def fn(self, v):
        self.ops.append((op, v, reverse))
        return self

    return fn


# Helper class for config items to reference other entries.
class Ref(object):

    def __init__(self, name):
        self.name = name
        self.ops = []

    def compute_value(self, value):
        for op, v, reverse in self.ops:
            if reverse:
                value = op(v, value)
            else:
                value = op(value, v)
        return value

    def __str__(self):
        raise NotImplementedError('Cannot convert Ref to string')

    # Supported operations.
    __add__ = _binary_op(operator.add)
    __radd__ = _binary_op(operator.add, reverse=True)
    __mul__ = _binary_op(operator.mul)
    __rmul__ = _binary_op(operator.mul, reverse=True)
    __sub__ = _binary_op(operator.sub)
    __rsub__ = _binary_op(operator.sub, reverse=True)
    __mod__ = _binary_op(operator.mod)
    __truediv__ = _binary_op(operator.truediv)
    __rtruediv__ = _binary_op(operator.truediv, reverse=True)
    __floordiv__ = _binary_op(operator.floordiv)
    __rfloordiv__ = _binary_op(operator.floordiv, reverse=True)
    __getitem__ = _binary_op(operator.getitem)


def process_config_files(config_file_paths, default_ext='.txt'):
    """Finds and processes config files in the given list of paths.

    Args:
        config_file_paths: List of paths. Globs (*) will automatically be
            expanded. Directories will be searched for files of the config
            extension type.
        default_ext: The default extension type of configs. Directory paths will
            be searched for this extension type.
    Returns:
        A list of expanded configurations from the files.
    """
    paths = []
    # Expand globs and directories.
    for path in config_file_paths:
        if '*' in path:
            paths.extend(glob.glob(path))
        elif os.path.isdir(path):
            paths.extend(glob.glob(os.path.join(path, '*' + default_ext)))
        else:
            paths.append(path)
    # Add jobs from config files.
    configs = []
    for path in sorted(paths):
        path_configs = process_config_file(path)
        configs.extend(path_configs)
        print('Added {} configs from {}'.format(len(path_configs), path))
    return configs


def process_config_file(config_file_path):
    """Returns a list of expanded configurations from the given file path."""
    with open(config_file_path, 'r') as file:
        return process_config(file.read())


def process_config(config_str):
    """Returns a list of expanded configurations from the given string."""
    config_value = eval(config_str)
    return expand_configs(config_value)


def expand_configs(config_dict):
    """Expands the config dictionary and resolves any references."""
    configs = expand_value(config_dict)
    configs = [resolve_references(c) for c in configs]
    return configs


def expand_value(value):
    """Expands the given configuration value. All lists in the value are
    flattened, and a cross-product of the lists are returned.
    """
    if not isinstance(value, list) and not isinstance(value, dict):
        return value
    elif isinstance(value, list):
        # Treat 1-length arrays as scalars.
        if len(value) == 1:
            return expand_value(value[0])
        # Return a flattened list of expanded subitems.
        expanded_list = []
        for v in value:
            v = expand_value(v)
            if isinstance(v, list):
                expanded_list.extend(v)
            else:
                expanded_list.append(v)
        return expanded_list
    # We have a dict.
    keys = value.keys()
    vals = []
    for v in value.values():
        v = expand_value(v)
        vals.append(v if isinstance(v, list) else [v])
    zipped_vals = list(zip(keys, v) for v in itertools.product(*vals))
    return [dict(z) for z in zipped_vals]


def resolve_references(config):
    """Resolves any Ref values in the given config."""
    config = copy.deepcopy(config)
    key_paths = {}

    def get_paths(d, prefix=''):
        for key, val in d.items():
            path_key = prefix + '.' + key if prefix else key
            if key in key_paths:
                print('WARNING: Duplicate path key: ' + path_key)
                continue
            if isinstance(val, dict):
                get_paths(val, prefix=path_key)
            else:
                key_paths[path_key] = (d, key, val)

    get_paths(config)

    unresolved_keys = set(key_paths.keys())
    resolved_keys = set()

    def resolve(key):
        parent_dict, val_key, val = key_paths[key]
        assert val_key in parent_dict
        if key in unresolved_keys:
            unresolved_keys.remove(key)

        # Ensure the key hasn't already been seen.
        if key in resolved_keys:
            raise KeyError('Cyclic key in config: ' + key)
        resolved_keys.add(key)

        if not isinstance(val, Ref):
            return
        if val.name not in key_paths:
            raise KeyError('Non-existent key in config: ' + val.name)

        # If the referenced value is also a Ref, defer evaluation until
        # later. Otherwise, change the value.
        _, _, ref_val = key_paths[val.name]
        if isinstance(ref_val, Ref):
            resolve(val.name)
            _, _, ref_val = key_paths[val.name]

        assert not isinstance(ref_val, Ref)
        result_val = val.compute_value(ref_val)
        parent_dict[val_key] = result_val
        key_paths[key] = (parent_dict, val_key, result_val)

    while len(unresolved_keys):
        resolve(unresolved_keys.pop())

    return config
