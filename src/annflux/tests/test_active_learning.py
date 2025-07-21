# Copyright 2025 Intel Corporation
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
from __future__ import annotations

import unittest


from annflux.tools.data import compute_incorrect_score


class TestIncorrectScore(unittest.TestCase):
    def test_computation(self):
        self.assertAlmostEquals(
            0.33,
            compute_incorrect_score(
                "vegetative",
                "blurry-or-low-res,flowering,low-quality",
                "0.13",
                "blurry-or-low-res,flowering,low-quality",
                "0.94,0.87,0.86",
            ),
        )
