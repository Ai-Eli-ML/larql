# HEAD-OBS-1 — which heads write the answer transition, and where do they read from?

Pre-registered 2026-09-20, before any per-head data exists on this executor. Laws, witness
sites, questions and forecasts below are FROZEN. Nothing here claims a result, and nothing
here claims causality.

Programme: OBSERVE. Rung above V3-OBS-1 (#484), V3-STREAM-1 (#485), the `observe` verb
(#486) and V3-LENS-1 (branch `lens-1`); the executor side of the Observatory's *Rich*
capture profile. Earned by INSTRUMENT-1a (`instrument-1-calibration.md`, DB
`instrument-1a-edge1-calibration`), whose disagreement ledger this rung exists to
adjudicate.

---

## The question

INSTRUMENT-1a established, on the sealed EDGE-1 bank, that the counterfactual log-odds
shift toward the rewritten answer enters the answer position's carrier in **one attention
write** — L23 on Gemma 3 4B, L35 on 12B — and is amplified at a second, L29 / L41. It also
established that Anatomist's hottest direct-logit-attribution head is not that write's
author: it is a late head (L31 H7 / L44 H10, henceforth the **late high-DLA head**, a
neutral name) whose direct contribution tracks whichever label is already favoured, and on
12B the raw-row attribution is dominated by layer 0–2 artefacts.

Two things are therefore unknown and cannot be learned from any existing instrument:

1. **Writer identity.** Within the emergence write, which query heads carry the reader
   delta, and how concentrated is it?
2. **Information source.** Where do those heads read from? L23 is the first *large*
   answer-position write; it is not necessarily where the relation was computed. Attention
   moves information across positions, so the write's content was assembled elsewhere and
   earlier. Only the heads' actual source distributions can say whether they read the
   edited relation-bearing line, the start node's mention in the question, or an
   already-resolved representation.

And a third, quieter one: INSTRUMENT-1a's frozen first-divergence marker fired *before*
the emergence write in most cases (a correctly signed, persistent, sub-nat to 3-nat
separation). That precursor is prior evidence now, and this rung asks whether it comes
from the same heads and sources as the emergence or from a different stage.

HEAD-OBS-1 answers these by **observation only**: per-head output contributions and
per-head source distributions, recorded as children of the attention writes V3-OBS-1
already records, under the same parity law.

---

## What already exists (read before implementing)

| Fact | Where |
|---|---|
| The attention step returns `AttentionStepOut { key, value, output }` with `output` **post gate, post output-projection**; per-head context never surfaces. V3-OBS-1 said so and refused to infer it. | `opplan/exec/backend.rs` (`AttentionStepCall`/`AttentionStepOut`); `observe.rs` ("`o_proj` … never surfaces"); `v3-obs-1-carrier-observation.md` §"Per-head observation is a separate rung" |
| Production: `project_position` → `(q, k, v, gate)`, then `attend_position(call, position, q, keys, values, pre, gate)` → `output`. Reference and device have their own `attention_step`. | `opplan/exec/production.rs:1046`, `reference.rs:727`, `device.rs:549` |
| The call carries everything the bound operation needs: `num_q_heads`, `num_kv_heads`, `head_dim`, `w_o`, `qk_norm`, `query_scale`, `score_scale`, `logit_softcapping`, `span`, `window`, `gate`, `bias` (O bias after the projection), `sinks`. | `backend.rs` `AttentionCall` |
| After the step, decode applies the family's **post-attention norm** (`state.post_attention`, Gemma 3 has one; Granite/OLMo do not), then `scale_residual_delta`, then `leave_site` → the recorded `CarrierWriteRecord { delta, after, layer_scale }`. | `opplan/exec/decode.rs` (attention site, `leave_site`) |
| Gemma 3 4B: 8 query heads over 4 KV heads, head_dim 256, score scale 1/16, per-head QK-norm with weight offset 1, no softcap, no gate, no bias, no sinks; **full-span attention at layers 5, 11, 17, 23, 29**, sliding (window 1024) elsewhere. 12B: 16 over 8, full-span at 5, 11, 17, 23, 29, 35, 41, 47. | container `system_graph.json` (`components[0].attention[*].span`, `execution.attention`) |
| The observer contract: structural `StepEvent`s plus borrowed-record taps (`carrier_write`, `entering_carrier`, …); recorder assigns sequence; receipt hashes the log. | `observe.rs`, `larql-inference/src/vindex3/record.rs` |

A structural fact worth stating before any result: **every emergence and amplification
write INSTRUMENT-1a found is a full-span attention layer** (23, 29 on 4B; 35, 41 on 12B),
and the two largest precursors on 12B sit at 29 (full-span). The 89-token prompt lies inside
the 1024 window, so sliding layers see the whole prompt too; the difference is the position
encoding (global θ with the linear factor) and what those layers learned. This is read off
the plan, not the record, and it is a fact the witnesses below will be reported against,
not a forecast.

---

## Contract laws (frozen as properties)

Let one attention write at `(layer l, position p)` have query heads `h = 0..H_q−1`, each
bound to KV head `g(h) = h ÷ (H_q / H_kv)`, context `ctx_h ∈ ℝ^{head_dim}` (the softmax-
weighted value sum), and the executor's own output `o ∈ ℝ^{hidden}`.

**HL1 — head-sum law.** The observer computes, beside the executor's fused output and never
in place of it, per-head output contributions `c_h = W_o[:, h·d..(h+1)·d] · ctx_h` (the
family's gate, if any, applied where the plan applies it, per head when it is elementwise on
the concatenated context) and declares the once-only terms (O bias). Property: `‖Σ_h c_h +
bias − o‖ / ‖o‖ ≤ 1e−4` at every observed write, on every backend, with the measured value
recorded; bit identity is **not** required across accumulation orders and is not claimed.
The **children of the recorded write** are `c′_h = s · (1 + w_post) ⊙ c_h · layer_residual_scale`
where `s = 1/rms(o + bias)` is the post-attention norm's scalar for this write (`1` and no
gain when the family has no post-attention norm), so that `Σ_h c′_h + bias′ = delta` exactly
up to the same tolerance, where `delta` is the delta V3-OBS-1 already records. The reader-
projected form `Σ_h ⟨r, c′_h⟩ = ⟨r, delta⟩` follows by linearity and is checked too.

**HL2 — source law.** Each head record carries its actual post-softmax attention
distribution over the positions the bound operation attended: `0..=p` for full span, the
window's positions for a sliding span (positions outside the span are **absent**, never
zero), plus the sink mass when the plan declares sinks. `Σ probs + sink = 1` within 1e−6.
The record carries the KV head index `g(h)` the query head was bound to; for GQA/MQA the
distribution is the query head's, over the shared KV head's rows — the bound operation,
not a transformer-cartoon abstraction. No "source token" is inferred; positions map to
token ids through the record's own token list.

**HL3 — parity law.** With head observation armed on every site, logits at every position,
every recorded carrier write (`delta`, `after`, `layer_scale`), every structural event and
the run's provenance are bit-identical to the unobserved run on the reference and
production CPU backends. The observer reads `ctx_h` and the probabilities the executor
already computed and computes `c_h` separately; the executor's arithmetic is untouched.
Observation is subscription, never a second executor.

**HL4 — accounting law.** The site-level attention write stays authoritative. Head
observations are children keyed to `(run, position, layer, site = attention)`; a record
without them is still a complete V3-OBS-1 record. A backend or path that cannot provide the
decomposition (the Metal whole-stack path, the batch prefill path) records
`head_observation: refused { reason }` for that write. Nothing reconstructs a head
decomposition from `delta` after the fact.

**HL5 — capture is declared and priced.** Two levels, chosen per run: *stats* at every
attention write (per head: `‖c′_h‖`, its projection on the run's basis, `g(h)`, the top-`k`
source positions with their probabilities and the sink mass, `k` declared) and *full* at
armed sites only (the whole source distribution and, on request, `c′_h` itself). The cost
class is one `W_o` slice product per head per observed write — the same multiply-adds as
one `o_proj` — plus the probabilities the executor already holds. Measured, not described:
token wall with stats-level head observation on every site minus `NoopObserver`, release,
production CPU, Granite 4.2 3B and Gemma 3 4B; per token and per write. Not a performance
claim.

**HL6 — on the record.** Head records are events with their own sequence, replay equal,
receipt covering them; the Standard adapter's "attention source capture unavailable" flag
turns into a declared per-record capability rather than a constant.

---

## Frozen acceptance properties (engineering)

- **HP1** HL1 on the golden plan (both backends), on Granite 4.2 3B (no post-attention norm:
  the `s = 1` branch), on Gemma 3 4B (post-attention norm branch), every write of an 8-token
  run; the maximum relative residual reported per subject.
- **HP2** HL2: sums within 1e−6 at every write; sliding-span writes on Gemma 3 at a position
  beyond the window (a 1100-token synthetic prompt on the golden plan's geometry, or a real
  one if cheap) carry exactly the window's positions.
- **HP3** HL3 bit-parity, both CPU backends, golden and Gemma 3 4B.
- **HP4** HL4: the device path refuses with a reason; a record with refusals still replays.
- **HP5** HL5 measured on both real subjects.
- **HP6** HL6 replay identity and receipt coverage; the run record schema gains one event
  kind (`head_write`) and one refusal spelling, nothing else (forecast; if more is needed it
  is recorded as a finding).
- **Unwitnessed by this rung, declared:** families with an attention output gate (Qwen 3.8
  is on disk but large; not run), with O bias, with sinks (no container on disk) — the law's
  terms for them are written, not exercised.

---

## Scientific witnesses (predeclared; no case is selected after looking at head data)

Prompts, readers and arms are INSTRUMENT-1a's, unchanged (`chris-experiments/larql/
D_instrument1_edge1_calibration/{prompts,readers,readers12b}`). Per-head reader
contribution at a write is `⟨r_B − r_A, c′_h⟩ / rms(after)` in the same units as
INSTRUMENT-1a's `Δ`; the per-head **counterfactual contribution** is its target-arm value
minus its base-arm value, in the units of `S`. The site's own `S` step at that write is the
denominator for every "fraction" below and is already on record:

| subject | write | cases (S step at the write, target − base, nats) |
|---|---|---|
| 4B | **L23 attention (emergence)** | g00 11.6, g12 18.7, g13 13.0, g20 27.6, g22 19.5, g26 17.3 (the six reproducing); g14 17.2, g15 16.8 (target flips, base correct, a control arm failed G3 — secondary) |
| 4B | **L29 attention (amplification)** | g00 13.8, g12 16.2, g13 21.9, g20 24.6, g22 13.5, g26 14.2 |
| 4B | L31 attention (the late high-DLA head's site) | all six |
| 12B | **L35 attention (emergence)** | g26 18.3, g03 10.2, g00 11.2 |
| 12B | **L41 attention (amplification)** | g26 45.4, g03 22.0, g00 29.8 |
| 12B | L44 attention (late high-DLA head's site) | the three |

**Witness A — emergence.** For every case and both subjects, at the emergence write and
the amplification write, record per head: `‖c′_h‖`, counterfactual contribution, fraction
of the site's `S` step, `g(h)`, the full source distribution in base and target arms, and
the top five source positions with their tokens. Then, per write:

- **A1 concentration.** Fraction of the site's `S` step carried by the top head, and the
  number of heads needed to reach 90% of it.
- **A2 source.** For the top head in the target arm: attention mass on (i) the rewritten
  label token itself, (ii) the rest of the edited line, (iii) the start node's mention in
  the question line, (iv) the query's relation word in the question, (v) the other edge
  lines, (vi) BOS. These six classes are fixed now from the prompt structure; a position
  belongs to exactly one.
- **A3 identity across arms.** Is the top head toward B in the target arm the same head as
  the top head toward A in the base arm at that write? Same across the six cases?
- **A4 the late high-DLA head.** Its counterfactual contribution and source classes at its
  own site, reported beside the emergence heads, under its neutral name.

**Witness B — precursor.** Prior evidence from INSTRUMENT-1a's frozen `k*` (target arm,
first persistent divergence outside the control envelope): 4B g12 **L17 attention** (full
span, S 0.15), g13 L21 attention (0.32), g00 L15 ffn / g20 L20 ffn / g26 L20 ffn (the
attention write of the same layer is the witness, and the ffn write's own reader delta is
reported beside it), g22 none before L23; 12B g26 and g00 **L29 attention** (full span,
S 1.8 and 3.0, the L29 attention steps being 1.8 and 2.9 nats), g03 L26 attention (0.14).
For each, the same per-head record as Witness A. Question **B1**: does the precursor's
counterfactual contribution sit in the same head index and read from the same source
classes as the emergence head(s) of that case, or in different heads and classes? **No
story is pre-registered as the answer.** The two candidates the source law discriminates
are written down so the reading cannot drift: (a) the emergence head copies the rewritten
label from the edited line (class i/ii dominant), the precursor being an earlier, weaker
read of the same line; (b) the relation was resolved earlier at the start node's position
and the emergence head reads that resolved state from the question line (class iii
dominant), the precursor being the resolution stage leaking into the answer position.
Either, both in different cases, or neither is an admissible result.

---

## Forecasts (falsifiable; deliberately few)

- **HF1** HL1 residual ≤ 1e−5 relative on both CPU backends (f32 accumulation order only).
- **HF2** HL5 stats-level cost on Gemma 3 4B ≤ 15% of token wall (one extra `o_proj`-class
  product per attention write against a forward dominated by FFN and the head).
- **HF3 (A1)** the emergence write's `S` step is carried by **at most two heads for 90%** in
  at least 4 of the 6 reproducing 4B cases. Prior: weak; DLA's layer-sum at L23 was small
  relative to the reader's step, which is compatible with either one dominant head whose
  contribution DLA under-reads or several heads.
- **HF4 (A2)** the top head's source mass is **concentrated**: its top three positions hold
  at least half of the non-sink mass in at least 4 of 6 cases. No forecast on *which*
  class; that is the question.
- **HF5 (A3)** the top head is the same head index in base and target arms in at least 4
  of 6 cases (a label-reading head, not a B-specific one). No forecast across scale.
- **HF6 (B1)** no forecast.

---

## What is not claimed

Nothing causal. A head carrying 90% of the emergence step while attending to the rewritten
label is a **localised observed contribution and source pattern**, not necessity and not
sufficiency; a head with a large contribution may be compensated, and a head with a small
one may be required. The words *decider*, *amplifier*, *transporter/copier* and
*redundant contributor* are reserved for the intervention rung that can distinguish them
(zero the candidate emergence head; zero the late high-DLA head; compare the effect on
emergence and on final amplification), and are not used for any head in this rung's
results. Nothing about layers whose attention this rung does not observe (the Metal path,
batch prefill). Nothing about Gemma 3 beyond this bank and format.

## Out of scope

Interventions of any kind; the Metal whole-stack path (refusal only); the batch prefill
path; per-neuron FFN decomposition; the Observatory UI's Rich rendering (it consumes the
record; owed separately); any change to V3-OBS-1's carrier record or V3-LENS-1's lens
events beyond adding the head event kind.

## Record-keeping

Registered in the chuk-experiments DB (programme `larql`) with this document's sha256
before implementation; results appended below the forecasts after they are frozen;
per-case per-head tables and source distributions in the INSTRUMENT-1 bundle directory
under `head-obs-1/`. Order: golden and Granite engineering witnesses → Gemma 3 4B
engineering witnesses → Witness A on 4B → Witness B on 4B → 12B replay of both.

## Verdict rule

HEAD-OBS-1 is complete when HP1–HP6 pass, Witness A is recorded for the six reproducing 4B
cases at both writes and for the three 12B cases, Witness B is recorded for every case with
a defined precursor, and every head in the results is named by index and site only. It
unlocks INTERVENE-1 with a *named* candidate set: the emergence heads, the precursor heads
and the late high-DLA head, each with a recorded source pattern to be tested rather than
a role to be assumed.
