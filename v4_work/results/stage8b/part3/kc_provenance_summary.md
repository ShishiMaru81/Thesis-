# Kc provenance — summary

- Source checked: http://www.fao.org/4/x0490e/x0490e0b.htm (fetched 2026-09-26) — FAO-56 Table 12, Allen/Pereira/Raes/Smith 1998.
- 87 crop x stage rows total (29 crops x 3 stages).
- 9 rows (3 crops: Aman/Aus/Boro Rice) are cited to Islam (2021), not FAO-56 — not checked against this table (out of scope for this sheet).
- 78 rows carry an FAO-56 citation and were checked against the live-fetched table.
  - **43 match** FAO-56 Table 12 (a single printed value, or its explicitly-named group default) once the closest printed analog is used where no literal crop row exists.
  - **2 fall inside a printed RANGE** rather than a single figure (Cotton's kc_mid '1.15-1.20' and kc_end '0.70-0.50') — the CSV value equals a literal endpoint of the range in both cases, not an interpolated or averaged number.
  - **3 DIFFER** from the printed FAO-56 value even after using the closest analog: Brinjal/mid, Brinjal/end, Onion/ini.
  - **30 have no FAO-56 value at all** (crop not in Table 12, and for 5 crops — Guava, Mango, Litchi, Jackfruit, Jujube — no analog of any kind exists in the table): 30 rows across Mustard (3), Jute (3), Turmeric/Ginger/Mukhi Kachu (9, all matching only the Roots & Tubers group default, not a crop-specific figure), Guava/Mango/Litchi/Jackfruit/Jujube (15) — see the sheet's per-row notes for the exact breakdown.

## Headline findings for the team

1. **Citation label is wrong for all 26 FAO-56-cited crops**: the numbers are consistent with FAO-56 Table 12 (Allen et al. 1998), not "Doorenbos & Pruitt 1977" (FAO-24, a different document). Fix the label, not (necessarily) the numbers.
2. **Onion kc_ini = 0.15 does not match FAO-56** (dry onion's kc_ini is the Small Vegetables group default, 0.7) — confirms the outlier named in the Stage 8B prompt.
3. **Brinjal (Eggplant) kc_mid and kc_end do not match** FAO-56's Egg Plant row (CSV 1.15/0.80 vs FAO-56 1.05/0.90).
4. **Jute's values are identical to this CSV's own Cotton row** and Jute is not in FAO-56 at all — looks like an undisclosed substitution rather than a sourced figure.
5. **Five tropical fruit crops (Guava, Mango, Litchi, Jackfruit, Jujube) have no FAO-56 Table 12 entry and no analog** in the table (not even a generic "Fruit Trees" row) — their citation is unverifiable against this source by construction, not just unchecked.
6. Turmeric, Ginger and Mukhi Kachu's values equal the Roots & Tubers group default exactly — plausible, but not evidence of a crop-specific source.
7. Coconut (via "Palm Trees") and Lemon (via the named 70%-canopy citrus row) match exactly and are the strongest-sourced entries in the sheet.

kc_values_hybrid.csv was not modified.
