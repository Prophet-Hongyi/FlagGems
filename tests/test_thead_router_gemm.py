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

import pytest
import torch

import flag_gems


pytestmark = [
    pytest.mark.router_gemm,
    pytest.mark.skipif(
        flag_gems.vendor_name != "thead",
        reason="T-Head split-K dispatch is backend-specific",
    ),
]


def test_thead_router_gemm_splitk_policy():
    from flag_gems.runtime.backend._thead.ops.mm import router_splitk_scenario

    bf16 = torch.empty((1, 1), dtype=torch.bfloat16)
    fp16 = torch.empty((1, 1), dtype=torch.float16)

    assert router_splitk_scenario(bf16, bf16, 1024, 8, 4096)
    assert not router_splitk_scenario(fp16, fp16, 1024, 8, 4096)
    assert not router_splitk_scenario(bf16, bf16, 2048, 8, 4096)
    assert not router_splitk_scenario(bf16, bf16, 1024, 2048, 4096)
    assert not router_splitk_scenario(bf16, bf16, 1024, 8, 3840)


ROUTER_SPLITK_SHAPES = [
    (1, 256, 7168),
    (8, 256, 7168),
    (16, 256, 7168),
    (32, 256, 7168),
    (128, 256, 7168),
    (1, 64, 4096),
    (16, 128, 4096),
    (64, 384, 7168),
    (1024, 8, 4096),
]


@pytest.mark.parametrize("M,N,K", ROUTER_SPLITK_SHAPES)
def test_thead_router_gemm_splitk_accuracy(M, N, K):
    assert flag_gems.router_gemm.__module__ == "_thead.ops.mm"
    torch.manual_seed(20260815)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=flag_gems.device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=flag_gems.device)
    ref_out = torch.mm(x.cpu().float(), weight.cpu().float().t())

    with flag_gems.use_gems():
        out = flag_gems.router_gemm(x, weight)

    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), ref_out, rtol=1e-3, atol=1e-2)
