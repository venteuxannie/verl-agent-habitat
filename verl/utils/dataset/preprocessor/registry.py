# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Register:
    """Registry for preprocessor classes."""

    def __init__(self, name):
        self._name = name
        self._obj_dict = dict()

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, keys={list(self._obj_dict.keys())})"

    def _register(self, name, obj):
        if name in self._obj_dict:
            raise KeyError(f"Object '{name}' is already registered in '{self._name}'")
        self._obj_dict[name.lower()] = obj

    def register(self, obj=None):
        """Register a preprocessor class.
        
        Can be used as a decorator or function call.
        """
        if obj is None:  # Used as a decorator

            def deco(func_or_class):
                self._register(func_or_class.__name__, func_or_class)
                return func_or_class

            return deco
        # Used as a function call
        self._register(obj.__name__, obj)
        return obj

    def get(self, name):
        """Get a preprocessor class by name."""
        obj = self._obj_dict.get(name.lower())
        if obj is None:
            raise KeyError(f"Object '{name}' not found in '{self._name}' registry.")
        return obj


PREPROCESSOR_REGISTER = Register("Preprocessor Register")



