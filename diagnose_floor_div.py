import torch

import flag_gems


shape = (1024, 1024)
for attempt in range(10):
    lhs = torch.randn(shape, dtype=torch.float16, device=flag_gems.device)
    rhs = torch.randn(shape, dtype=torch.float16, device=flag_gems.device)
    rhs = rhs + torch.sign(rhs).clamp(min=1) * 1e-3
    lhs_cpu = lhs.cpu()
    rhs_cpu = rhs.cpu()
    print("result_type", torch.result_type(lhs, rhs), "gems_fn", flag_gems.ops.div_mode.__module__)
    reference = torch.ops.aten.div.Tensor_mode(
        lhs_cpu, rhs_cpu, rounding_mode="floor"
    )
    baseline = torch.ops.aten.div.Tensor_mode(lhs, rhs, rounding_mode="floor").cpu()
    actual = flag_gems.ops.div_mode(lhs, rhs, rounding_mode="floor").cpu()
    bad = (actual != reference).ravel().nonzero().ravel()
    if bad.numel():
        flat = bad[:50]
        a = lhs_cpu.ravel()[flat]
        b = rhs_cpu.ravel()[flat]
        print("attempt", attempt, "mismatches", bad.numel())
        print("lhs", a.tolist())
        print("rhs", b.tolist())
        print("actual", actual.ravel()[flat].tolist())
        print("reference", reference.ravel()[flat].tolist())
        print("baseline", baseline.ravel()[flat].tolist())
        print("devices", lhs.device, lhs_cpu.device, reference.device)
        print("baseline mismatches", (baseline != reference).sum().item())
        recomputed = torch.ops.aten.div.Tensor_mode(
            lhs_cpu, rhs_cpu, rounding_mode="floor"
        )
        print("recomputed", recomputed.ravel()[flat].tolist())
        print("ref/recomputed mismatches", (reference != recomputed).sum().item())
        print("ptrs", reference.data_ptr(), baseline.data_ptr(), actual.data_ptr(), recomputed.data_ptr())
        print("float32 floor", torch.floor(a.float() / b.float()).tolist())
        print("float64 floor", torch.floor(a.double() / b.double()).tolist())
        break
