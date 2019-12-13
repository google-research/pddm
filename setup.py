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

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pddm",
    version="0.0.1",
    author="Anusha Nagabandi",
    author_email="anagabandi@google.com",
    description="Code implementation of PDDM algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="todo",
    packages=['pddm'],
    classifiers=[],
)
