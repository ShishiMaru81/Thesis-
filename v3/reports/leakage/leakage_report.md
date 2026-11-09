# GATE 3 — Leakage Audit Report

- Generated: `2026-08-27T07:54:47.048900+00:00`
- Normalizer: `v3.1`
- Train: `D:\Research Lab\Thesis\v3\data\augmented\train_augmented.jsonl` (n=3552, sha256=f4ba6742198d7e1d…)
- Val: n=88  |  Test: n=182  (test sha256 verified=True)
- **Overall verdict: PASS**

## Gate table

| Detector | Threshold | Finding | Verdict |
|----------|-----------|---------|---------|
| EXACT | n/a | count=0 | PASS |
| JACCARD | 0.6 | pairs≥τ=4867, FAIL-rule hits (refined)=691 | PASS |
| SEMANTIC | 0.92 | candidates=794, pending_review=794, human_leakage=0 | PASS |
| SCENARIO | n/a | overlap=0 / heldout_hashes=25 | PASS |

### Detector 3 human-review note

> Pairs >= 0.92 are candidates for human review, not automatic FAIL. Write decisions into semantic_human_decisions.json. decision='leakage' fails this detector; 'legitimate' clears the pair.

Decisions file: `D:\Research Lab\Thesis\v3\reports\leakage\semantic_human_decisions.json`

## Detector 1 — exact matches (top 20)

_None._

## Detector 2: Lexical Overlap (Char 5-gram Jaccard)

**Threshold:** 0.60  
**Total pairs ≥ threshold:** 4,867  
**FAIL-rule hits (original):** 1,292  
**FAIL-rule hits (refined):** 691  
**Excluded (new qtype gate):** 601  

### Refined FAIL Rule

```
FAIL if: Jaccard >= 0.60 AND fact_id_train == fact_id_test AND qtype != "abstain"
```

All other high-Jaccard pairs (template similarity, different crops, shared abstain boilerplate) are **PASS**.

### FAIL-rule breakdown

| Kind | n | Interpretation |
|------|---:|----------------|
| fact_only_F3 | 691 | Synthetic numeric train shares `fact_id` with held-out numeric; scenario_hash still disjoint (Detector 4 PASS) |
| gold_only_boilerplate | 601 | Abstain pairs excluded by qtype gate — shared canned answer, different crops/facts |
| both | 0 | — |

### Threats and reasoning

**F3 fact-level leakage (691 pairs)**  
Numeric training rows are synthetic scenarios using all 84 crop-stage facts, including 25 facts appearing in val/test.  
Example: Litchi (initial stage) appears in both train synthetic (ET0=8.0, rain=5.5) and test seed (ET0=7.5, rain=6.2).  
Accepted because scenario-level separation is perfect (Detector 4 = 0 overlap). Model must apply formula to unseen weather, not memorize.  
Mitigation: slot-level F1 evaluation (Stage 6) by crop+stage.  
Verdict: documented as limitation in Stage 8 Threats section; not remediated.

**Gold-only false positive (601 pairs, excluded)**  
Abstain questions share canned boilerplate answer (“Model does not know this”). Two OOD questions with different crops hit high Jaccard.  
Example: Wheat (pest brand) vs Mango (price) — both abstain, both hit same gold answer, high template similarity.  
Excluded because abstain is a decision policy, not knowledge leakage.

### Top 20 Jaccard pairs (≥ threshold)

| jaccard | refined fail? | same_fact | same_gold | held | train |
|--------|---------------|-----------|-----------|------|-------|
| 0.9512 | False (abstain excluded) | False | True | গম ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? | আম ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? |
| 0.9512 | False | False | False | আম ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? | গম ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? |
| 0.9184 | False | False | False | আখ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? | আম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9184 | False | False | False | আখ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? | গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9070 | False | False | False | তিল ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? | তুল ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? |
| 0.9070 | False (abstain excluded) | False | True | তিল ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? | কুল ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? |
| 0.9048 | False (abstain excluded) | False | True | গম ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? | আখ ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? |
| 0.9048 | False | False | False | আম ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? | মর ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? |
| 0.9048 | False | False | False | আখ ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? | মর ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? |
| 0.9048 | False | False | False | আখ ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? | গম ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত? |
| 0.9000 | False | False | False | আলু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | আম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | আলু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | মটর ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | আম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | মটর ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | তিল ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | আম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | তিল ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? |
| 0.9000 | False | False | False | আখ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল? | লাউ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল |
| 0.8889 | False (abstain excluded) | False | True | আম ফসলের আজকের বাজার দর কত? | গম ফসলের আজকের বাজার দর কত? |
| 0.8889 | False | False | False | আলু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল | মিষ্টি আলু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সং |
| 0.8864 | False (abstain excluded) | False | True | আদা ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? | তুলা ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি? |

### Jaccard nearest-neighbour distribution

n=270  min=0.056338  mean=0.711893  median=0.730046  p90=0.848542  max=0.95122

- [0.00,0.20): 6
- [0.20,0.40): 0
- [0.40,0.50): 5
- [0.50,0.60): 23
- [0.60,0.70): 66
- [0.70,0.80): 106
- [0.80,0.90): 55
- [0.90,0.95): 7
- [0.95,1.01): 2

## Detector 3 — semantic candidates (≥ 0.92)

- cosine=0.9988 decision=None key=`test:92ba40c6-7204-457f-a6a7-a300ee09aa63::7718744f-afa4-45c4-b778-3bf23924dce5`
  - held: মটর ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: পেঁয়াজ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9986 decision=None key=`test:80041dbd-a976-4dc9-aed7-640b156ae0b4::9d8fd499-a10c-4918-95d8-e49264ab8a22`
  - held: আমার মিষ্টি আলু ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.২ মিমি, বৃষ্টি হয়েছে ৩.৯ মিমি, জমির পরিমাণ ১.৬ শতক। আজ কত পানি সেচ দিতে হবে?
  - train: আমার মিষ্টি আলু ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.১২ মিমি, বৃষ্টি হয়েছে ৭.০২ মিমি, জমির পরিমাণ ১.৫৬ শতক। আজ কত পানি সেচ দিতে হবে?
- cosine=0.9985 decision=None key=`test:41a43b74-fdfa-4b43-b60c-19d02cdf0ada::7718744f-afa4-45c4-b778-3bf23924dce5`
  - held: লিচু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: পেঁয়াজ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9982 decision=None key=`test:41a43b74-fdfa-4b43-b60c-19d02cdf0ada::8dc38b60-74ee-42de-b0fb-b23d92623bdb`
  - held: লিচু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: লাউ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9981 decision=None key=`test:7adfbc22-422c-4f97-9353-cc4832156a5e::ce76ea0e-8f69-476a-8e87-22633e413454`
  - held: আমার খামারে আখ ফসলের সঠিক মাটির pH কত?
  - train: আমার খামারে আদা ফসলের সঠিক মাটির pH কত?
- cosine=0.9981 decision=None key=`test:92ba40c6-7204-457f-a6a7-a300ee09aa63::8dc38b60-74ee-42de-b0fb-b23d92623bdb`
  - held: মটর ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: লাউ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9981 decision=None key=`test:a0ad6d8e-7e0c-4858-b1a4-ab61c98651b2::aa1815ab-58ce-4a69-b8c1-963a8d0fd763`
  - held: আদা ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি?
  - train: পাট ফসলের জন্য সবচেয়ে সস্তা কীটনাশক ব্র্যান্ড কোনটি?
- cosine=0.9979 decision=None key=`val:e7dab608-8092-4684-8c92-adfb9e251ef3::170e3b9a-3805-467a-8054-ea9d33ce2157`
  - held: আমার খামারে পাট ফসলের সঠিক মাটির pH কত?
  - train: আমার খামারে কুল ফসলের সঠিক মাটির pH কত?
- cosine=0.9979 decision=None key=`test:50649cf0-be92-4889-86a4-e0b4c909851b::7718744f-afa4-45c4-b778-3bf23924dce5`
  - held: সরিষা ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: পেঁয়াজ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9979 decision=None key=`test:f773692a-2fda-4520-ba3c-a940f0beb3bf::8dc38b60-74ee-42de-b0fb-b23d92623bdb`
  - held: মুখীকচু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: লাউ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9979 decision=None key=`test:19b0ac34-567f-40cf-83aa-562a9d4c5df8::8dc38b60-74ee-42de-b0fb-b23d92623bdb`
  - held: আখ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: লাউ ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9979 decision=None key=`test:80041dbd-a976-4dc9-aed7-640b156ae0b4::3dc071ea-80dc-430c-b3a4-8463722d9d77`
  - held: আমার মিষ্টি আলু ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.২ মিমি, বৃষ্টি হয়েছে ৩.৯ মিমি, জমির পরিমাণ ১.৬ শতক। আজ কত পানি সেচ দিতে হবে?
  - train: আমার মিষ্টি আলু ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.৩৫ মিমি, বৃষ্টি হয়েছে ৭.৫ মিমি, জমির পরিমাণ ০.১২ বিঘা। আজ কত পানি সেচ দিতে হবে?
- cosine=0.9978 decision=None key=`test:a8aed623-a018-47a0-a50a-4fc97008945d::c18606e6-4513-410a-b970-46d0bc73f1a7`
  - held: আমার লাউ ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৮.১ মিমি, বৃষ্টি হয়েছে ২.৩ মিমি, জমির পরিমাণ ০.২১ কাঠা। আজ কত পানি সেচ দিতে হবে?
  - train: আমার তুলা ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৮.৩৯ মিমি, বৃষ্টি হয়েছে ১৮.৮ মিমি, জমির পরিমাণ ০.২ একর। আজ কত পানি সেচ দিতে হবে?
- cosine=0.9978 decision=None key=`test:1293a1af-0d26-46cf-90f2-4d15fe3f1e8e::2bc8fea6-b696-441c-9b6b-e57cebe304c8`
  - held: My Boro Rice field is at the mid-season stage. Today ET0 is 5.8 mm, it rained 32.0 mm, and the field is 0.39 hectare. How much irrigation water does it need today?
  - train: My Boro Rice field is at the mid-season stage. Today ET0 is 6.32 mm, it rained 13.39 mm, and the field is 0.82 hectare. How much irrigation water does it need today?
- cosine=0.9977 decision=None key=`test:1b826c69-1876-49b8-91d0-405a419514c4::c20d4c32-61ed-4d8d-86dc-3e7620ff52a7`
  - held: My Sweet Potato field is at the late-season stage. Today ET0 is 6.2 mm, it rained 3.9 mm, and the field is 1.6 decimal. How much irrigation water does it need today?
  - train: My Sweet Potato field is at the late-season stage. Today ET0 is 6.12 mm, it rained 7.02 mm, and the field is 1.56 decimal. How much irrigation water does it need today?
- cosine=0.9977 decision=None key=`test:a8aed623-a018-47a0-a50a-4fc97008945d::ec98bf4d-bac8-4042-8fa1-79942b920da8`
  - held: আমার লাউ ক্ষেত মধ্য-মৌসুম পর্যায়ে আছে। আজ ET0 ৮.১ মিমি, বৃষ্টি হয়েছে ২.৩ মিমি, জমির পরিমাণ ০.২১ কাঠা। আজ কত পানি সেচ দিতে হবে?
  - train: আমার পেঁয়াজ ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৮.৪ মিমি, বৃষ্টি হয়েছে ৩.২ মিমি, জমির পরিমাণ ০.১২ কাঠা। আজ কত পানি সেচ দিতে হবে?
- cosine=0.9977 decision=None key=`test:9a51ebcf-f79c-48a3-901a-ef7f2454be13::67af80ec-e5e3-44c5-8fb0-1ea3eaf04b5e`
  - held: পাট ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত?
  - train: আদা ফসলের ক্ষেত্রে পানি নিষ্কাশন সম্পর্কে কী করা উচিত?
- cosine=0.9976 decision=None key=`test:41a43b74-fdfa-4b43-b60c-19d02cdf0ada::ca2424c8-966b-4cef-a44e-e348372ed1bd`
  - held: লিচু ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9976 decision=None key=`test:2ec31899-c02a-4b52-9f50-d3d8afcc4d45::ca2424c8-966b-4cef-a44e-e348372ed1bd`
  - held: তিল ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
  - train: গম ফসলের কোন বৃদ্ধির পর্যায়গুলো পানির অভাবে বেশি সংবেদনশীল?
- cosine=0.9975 decision=None key=`test:690b5559-8fce-4336-82a3-33b04841ca11::5799437d-8f77-443d-86f9-824965a5d3fe`
  - held: আমার সরিষা ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.৬ মিমি, বৃষ্টি হয়েছে ২১.০ মিমি, জমির পরিমাণ ৬.০ একর। আজ কত পানি সেচ দিতে হবে?
  - train: আমার বোরো ধান ক্ষেত শেষ-মৌসুম পর্যায়ে আছে। আজ ET0 ৬.৬৬ মিমি, বৃষ্টি হয়েছে ৫.০ মিমি, জমির পরিমাণ ৬.৪ বিঘা। আজ কত পানি সেচ দিতে হবে?

### Semantic nearest-neighbour distribution

n=270  min=0.651103  mean=0.921718  median=0.974409  p90=0.996238  max=0.998788

## Detector 4 — scenario_hash overlaps

_None. Zero overlap._

---

**STOP. Overall verdict: PASS. Approved to proceed to Stage 4.**
