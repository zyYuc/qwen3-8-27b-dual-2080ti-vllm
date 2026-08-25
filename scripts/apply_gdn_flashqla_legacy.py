#!/usr/bin/env python3
"""Manually apply the 2 rejected GDN patch hunks to 0.27.1 source."""
import sys

filepath = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
with open(filepath, "r") as f:
    content = f.read()

# 1. Insert flashqla_legacy_chunk_gated_delta_rule function before fi_chunk_gated_delta_rule
func_code = '''
def flashqla_legacy_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """FlashQLA legacy GDN prefill for SM70/SM75 (2080 Ti)."""
    from flash_qla.ops.gated_delta_rule.legacy import (
        chunk_gated_delta_rule_fwd_legacy,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    scale = q.shape[-1] ** -0.5

    def run_one(start: int | None = None, end: int | None = None, state_idx: int = 0):
        seq = slice(start, end) if start is not None else slice(None)
        output, final_state = chunk_gated_delta_rule_fwd_legacy(
            q[:, seq].to(torch.float32).contiguous(),
            k[:, seq].to(torch.float32).contiguous(),
            v[:, seq].to(torch.float32).contiguous(),
            g[:, seq].to(torch.float32).contiguous(),
            beta[:, seq].to(torch.float32).contiguous(),
            scale,
            initial_state[state_idx : state_idx + 1].to(torch.float32).contiguous(),
        )
        return output.to(v.dtype), final_state

    if cu_seqlens is None:
        output, final_state = run_one()
        if output_final_state:
            return output, final_state
        return output, None

    cu = cu_seqlens.detach().cpu().tolist()
    if len(cu) == 2 and cu[0] == 0 and cu[1] == q.shape[1]:
        output, final_state = run_one()
        if output_final_state:
            return output, final_state
        return output, None

    if q.shape[0] != 1:
        raise NotImplementedError(
            "FlashQLA legacy ragged GDN prefill expects packed q/k/v "
            "with batch dimension 1"
        )
    if cu[0] != 0 or cu[-1] != q.shape[1]:
        raise ValueError("cu_seqlens must cover the packed GDN prefill tokens")

    num_sequences = len(cu) - 1
    if initial_state.shape[0] not in (1, num_sequences):
        raise ValueError(
            "initial_state first dimension must be 1 or match "
            "the number of packed GDN prefill sequences"
        )

    outputs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        if end <= start:
            continue
        state_idx = seq_idx if initial_state.shape[0] == num_sequences else 0
        seq_output, seq_final_state = run_one(start, end, state_idx)
        outputs.append(seq_output)
        final_states.append(seq_final_state)

    if not outputs:
        output = torch.empty(
            (q.shape[0], 0, v.shape[2], v.shape[3]),
            device=v.device,
            dtype=v.dtype,
        )
        final_state = initial_state[:0] if output_final_state else None
    else:
        output = torch.cat(outputs, dim=1)
        final_state = torch.cat(final_states, dim=0)

    if output_final_state:
        return output, final_state
    return output, None


'''

# Insert before def fi_chunk_gated_delta_rule
if "flashqla_legacy_chunk_gated_delta_rule" not in content.split("def fi_chunk_gated_delta_rule")[0]:
    content = content.replace(
        "def fi_chunk_gated_delta_rule(",
        func_code + "def fi_chunk_gated_delta_rule(",
        1
    )
    print("Inserted flashqla_legacy_chunk_gated_delta_rule function")
else:
    print("Function already exists, skipping insert")

# 2. Fix the return type signature of _resolve_gdn_prefill_backend
old_sig = '-> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:'
new_sig = '-> tuple[str, Literal["triton", "flashinfer", "cutedsl", "flashqla_legacy"]]:'
if old_sig in content:
    content = content.replace(old_sig, new_sig, 1)
    print("Fixed _resolve_gdn_prefill_backend return type signature")
else:
    print("Signature already patched or not found")

# 3. Insert flashqla_legacy check before flashinfer check
old_check = '    if backend in ["flashinfer", "auto"] and supports_flashinfer:\n        return backend, "flashinfer"'
new_check = '    if backend == "flashqla_legacy":\n        return backend, "flashqla_legacy"\n    if backend in ["flashinfer", "auto"] and supports_flashinfer:\n        return backend, "flashinfer"'
if old_check in content and 'if backend == "flashqla_legacy":' not in content.split(old_check)[0]:
    content = content.replace(old_check, new_check, 1)
    print("Inserted flashqla_legacy backend check")
else:
    print("flashqla_legacy check already present or anchor not found")

with open(filepath, "w") as f:
    f.write(content)

print("Manual patch applied successfully")
