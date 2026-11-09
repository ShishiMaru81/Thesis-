# Phi-4-mini adapter caveat (Stage 5)

Phi-4-mini differs architecturally from the other three base models in a way
that propagates into its LoRA adapter, and the difference is not something
configuration can equalise. Llama-3.1-8B, Mistral-Nemo, and Unsloth's Phi-3.5-mini
repackage all expose separate attention and MLP projections, so each receives the
standard seven-module adapter (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`). Phi-4-mini instead ships **fused** projections: a single
`qkv_proj` carrying query, key, and value, and a single `gate_up_proj` carrying the
MLP gate and up paths. Its adapter therefore targets four modules — `qkv_proj`,
`o_proj`, `gate_up_proj`, `down_proj` — which cover exactly the same computational
paths as the other models' seven, but through fused weight matrices. The practical
consequence is that a rank-16 LoRA on `qkv_proj` shares one rank-16 down-projection
across Q, K, and V, rather than giving each its own as happens when the projections
are separate. The resulting adapter has 23,068,672 trainable parameters against
Llama's 41,943,040, Mistral-Nemo's 57,016,320, and Phi-3.5-mini's 29,884,416.
**This is Phi-4-mini's architecture, not a defect or a misconfiguration** — targeting
the fused module names is the correct and only way to adapt those paths, and every
attention and MLP projection in the model does receive adaptation.

It is worth recording explicitly how this was found, because the failure mode is
silent. Applying the seven-module Llama list to Phi-4-mini does not raise an error:
`o_proj` and `down_proj` match, so PEFT proceeds and attaches a partial adapter,
while `q_proj`, `k_proj`, `v_proj`, `gate_proj`, and `up_proj` match nothing at all.
That configuration produced an 8,912,896-parameter adapter with **no adaptation of
attention QKV and no adaptation of the MLP gate/up path** — a number reproduced
exactly by 32 layers x [`o_proj` 16x(3072+3072) + `down_proj` 16x(8192+3072)].
Had it gone unnoticed, any finding that Phi-4-mini underperforms would have been a
statement about adapter placement rather than about the model. All twelve Stage 5
runs were trained with the corrected per-architecture targets, and `s5_train.py`
now calls `assert_targets_attached()` after `get_peft_model`, which hard-fails if
any requested target owns zero LoRA layers, so this class of silent mismatch cannot
recur on a future base model.

## For Stage 8 (Threats to Validity)

> Phi-4-mini adapter uses fused-QKV LoRA; not structurally identical to the other
> three models' adapters.

Expanded, for the body of that section: Phi-4-mini's fused `qkv_proj` and
`gate_up_proj` mean its rank-16 adapter shares a single down-projection across Q, K,
and V, whereas Llama-3.1-8B, Mistral-Nemo, and Phi-3.5-mini receive independent
rank-16 adapters on each separate projection. The four arms of the comparison are
therefore matched on LoRA rank, alpha, dropout, learning rate, schedule, and token
budget, but not on adapter parameterisation, which follows from each base model's
architecture. Comparisons involving Phi-4-mini should be read as comparing
model-plus-native-adapter, not as an ablation holding adapter structure fixed.

## Reference

| model | targets | modules | trainable params |
|---|---|---|---|
| mistral_nemo | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` | 7 | 57,016,320 |
| llama31_8b | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` | 7 | 41,943,040 |
| phi35_mini | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` | 7 | 29,884,416 |
| phi4_mini | `qkv_proj` `o_proj` `gate_up_proj` `down_proj` | 4 | 23,068,672 |

Phi-4-mini config: 32 layers, hidden 3072, intermediate 8192, `qkv_proj` output 5120
(3072 query + 1024 key + 1024 value, i.e. 8 KV heads at head_dim 128).

Verified on the meta device (real module graph, no weights, no GPU allocation)
before the corrected runs were launched; the same method reproduced the three
unfused models' parameter counts exactly, which is what makes the Phi-4-mini
figure trustworthy. See `reports/s5/preflight_models.json` and the twelve
`reports/s5/*_seed*.json` manifests, each of which records the
`lora_target_modules` and `lora_modules_attached` actually used.
