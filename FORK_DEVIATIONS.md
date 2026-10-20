# Fork deviations

Base: `apple-aiml-research/ml-rsi-inference-bench` at `a6b10af282014412b4eae14f3105d68555889821`.

This fork runs the H100 tasks on NVIDIA H200 (same sm_90 architecture, 141 GB instead of 80 GB).
Only GPU-name gates that have no upstream waiver are changed; scoring logic is untouched.

| File | Change | Reason |
|---|---|---|
| `tasks/support_gpt_oss_120b_4xh100_sglang/tests/config.json` | `infb.gpu_name` `H100` → `H200` | `tests/preflight.py` requires the name as a substring of every visible device and has no override; without it every run voids (`reward.txt = -1`). |
| `tasks/support_kimi_linear_48b_4xh100_vllm/tests/config.json` | same | same |
| `tasks/support_diffusiongemma_26b_a4b_2xh100_vllm/tests/config.json` | same | same |

The `optimize_*` verifiers already honour `INFB_PREFLIGHT_ALLOW_GPU_MISMATCH=1`; the `regression_*` and
`support_sd3_2xh100_sglang` verifiers do not check the device name. Those tasks are unchanged.

Scores measured on H200 are not interchangeable with the upstream H100 leaderboard.
