# Copyright 2026 FlagOS Contributors
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

from types import SimpleNamespace

from flag_gems.utils import triton_lang_helper


def test_missing_normcdfinv_uses_portable_fallback():
    module = SimpleNamespace()

    patched = triton_lang_helper._patch_missing_symbols(module, ("normcdfinv",))

    assert patched is module
    assert patched.normcdfinv is triton_lang_helper._fallback_normcdfinv
