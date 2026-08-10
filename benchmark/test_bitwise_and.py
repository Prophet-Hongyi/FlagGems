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

from . import base, consts, utils


@pytest.mark.bitwise_and_tensor
def test_bitwise_and():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.bitwise_and_tensor_
def test_bitwise_and_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="bitwise_and_tensor_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0x3F


@pytest.mark.bitwise_and_scalar
def test_bitwise_and_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="bitwise_and_scalar",
        torch_op=torch.bitwise_and,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def bitwise_and_scalar_input_fn(shape, cur_dtype, device):
    inp1 = base.generate_tensor_input(shape, cur_dtype, device)
    if cur_dtype == torch.bool:
        inp2 = True
    else:
        inp2 = 0x00FF
    yield inp1, inp2


@pytest.mark.bitwise_and_scalar_
def test_bitwise_and_scalar_():
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_",
        torch_op=lambda a, b: a.bitwise_and_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        input_fn=bitwise_and_scalar_input_fn,
        is_inplace=True,
    )
    bench.run()


def scalar_tensor_input_fn(shape, cur_dtype, device):
    scalar = 0x96 if cur_dtype != torch.bool else True
    inp = base.generate_tensor_input(shape, cur_dtype, device)
    yield scalar, inp


@pytest.mark.bitwise_and_scalar_tensor
def test_bitwise_and_scalar_tensor():
    benchmark_kwargs = {}
    torch_op = torch.bitwise_and
    if flag_gems.vendor_name == "kunlunxin":
        # The native Scalar_Tensor overload cannot materialize the Python
        # scalar. AND is commutative, so use Tensor_Scalar for its baseline.
        torch_op = lambda scalar, tensor: torch.bitwise_and(tensor, scalar)
        benchmark_kwargs["gems_op"] = flag_gems.bitwise_and_scalar_tensor
    bench = base.GenericBenchmark(
        op_name="bitwise_and_scalar_tensor",
        torch_op=torch_op,
        input_fn=scalar_tensor_input_fn,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        **benchmark_kwargs,
    )
    bench.run()
