# Stage 8B hardware check

Arm: `llama31_8b_seed42`
Sample: 50 rows, seed 42, stratified by qtype × lang
Decoding: greedy, max_new_tokens 512, INFER_MAX_SEQ_LEN 2048, batch_size 16

## Results

| Check | Value | Required |
|---|---:|---|
| Byte-identical outputs | 36/50 (72.0%) | ≥ 90% |
| Classification matches | 50/50 | all 50 |
| exact_0.1 matches | 11/12 compared | all compared |
| **Pass** | **NO** | |

Peak VRAM: 10.771 GB; wall 60.5 s.
Strata: {('abstain', 'bn'): 7, ('abstain', 'en'): 7, ('clarify', 'bn'): 6, ('clarify', 'en'): 6, ('factual', 'bn'): 6, ('factual', 'en'): 6, ('numeric_runtime', 'bn'): 6, ('numeric_runtime', 'en'): 6}

## Failures
- `7f34cbc8-91f8-4658-853e-ca399aecc2bf` clarify/en: byte=False class=answer→answer exact=None→None
- `8739a33b-0d19-4954-b778-19aec52301f6` factual/bn: byte=False class=answer→answer exact=None→None
- `75150278-22df-42f9-b165-af36b662b56c` factual/en: byte=False class=answer→answer exact=None→None
- `ac0a4d8a-03ca-496b-badb-42c7bd5b5707` abstain/bn: byte=False class=answer→answer exact=None→None
- `f4c5bc75-3c18-42ac-a7a7-f64a8649d905` numeric_runtime/bn: byte=False class=answer→answer exact=True→True
- `9eab0c0c-5d00-4d49-8e46-236c6c891e3b` numeric_runtime/en: byte=False class=answer→answer exact=False→True
- `c47fb2c2-28df-4cb5-a72f-732ddd62a3e0` numeric_runtime/bn: byte=False class=answer→answer exact=True→True
- `2b0242c5-9a69-43dd-a37c-cd53e384b49f` abstain/bn: byte=False class=answer→answer exact=None→None
- `e99aacd9-85d4-477f-8752-dd4f4deaad25` numeric_runtime/bn: byte=False class=answer→answer exact=True→True
- `9160c39e-0c0d-4c80-bf5c-07ffd39e56e5` clarify/en: byte=False class=answer→answer exact=None→None
- `7a07817c-8d37-43c1-9793-0fc12c47d87c` numeric_runtime/bn: byte=False class=answer→answer exact=False→False
- `81cda28c-93bb-4fd7-b04d-a67cc89f0478` numeric_runtime/en: byte=False class=answer→answer exact=True→True
- `a3f3ab38-a484-4981-9fc8-abc237a2fa85` clarify/bn: byte=False class=answer→answer exact=None→None
- `637c02f6-7fad-4003-9318-4bbd10c9404c` numeric_runtime/bn: byte=False class=answer→answer exact=False→False
