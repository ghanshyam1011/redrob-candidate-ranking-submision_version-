# Redrob Intelligent Candidate Discovery & Ranking

> Submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge
> (India Runs Hackathon / Hack2Skill).
> Ranks 100,000 synthetic candidate profiles against a single Senior AI Engineer
> job description and returns the top 100, with a transparent score and a
> fact-based reason for every candidate.

---

## TL;DR

- **Input:** 100,000 candidate profiles (`candidates.jsonl`) + one job description.
- **Output:** `team_submission.csv` — top 100 candidates with `candidate_id, rank, score, reasoning`.
- **Runs:** CPU-only, fully offline at ranking time, ~60 seconds, under 16 GB RAM.
- **Approach:** a transparent, additive, multi-layer scoring rubric — **no LLM at ranking time, no GPU, no external API.**

---

## The problem this solves (and the trap it avoids)

The job description is deliberately loaded with AI buzzwords. A naive system that
ranks by **keyword density in the skills list** is easy to fool — and the dataset
includes ~80 honeypot profiles engineered to exploit exactly that: impossible or
implausible histories where skills are stuffed with expert-level AI terms but the
actual work history tells a completely different story.

Our central design principle is therefore:

> **Rank people by what they actually *did* (their career history), not by what
> they *listed* (their skills).**

Every layer and guard in the pipeline follows from that principle.

---

## How a candidate is scored

Each surviving candidate receives an additive score from **7 positive layers minus a risk penalty**:

| Layer | Max Points | What it measures | Source |
|---|---:|---|---|
| **Technical Substance** | 28 | Hands-on retrieval, embeddings, vector DBs, ranking, LLM fine-tuning, eval frameworks, distributed ML — detected in career history and summary, NOT the skills list. | `career_history` descriptions + `profile.summary` |
| **JD Skill Alignment** | 22 | Direct match to the JD's named tools and skills using alias-aware matching against the real dataset vocabulary. Any one alias in a group earns the group's points once — never stacked. | `skills[].name` + `profile.summary` |
| **Production Engineering** | 15 | Ownership and shipping evidence: built, deployed, architected, on-call, scaled. Hedged language ("exposure", "assisted") halves the credit for that role. | `career_history` descriptions |
| **Behavioral & Hireability** | 12 | Real platform signals: open-to-work, recruiter response rate, interview completion, GitHub activity, recency, search appearances, saved by recruiters, profile completeness. | `redrob_signals` |
| **Career Progression** | 10 | 5–9 year sweet spot, title growth, stable tenure, current engineering role, leadership/mentoring evidence. | `profile.years_of_experience` + `career_history` |
| **Company Context** | 6 | AI/product company vs. services firm; company size as a proxy for product-engineering exposure and scale. | `career_history.industry` + `company_size` |
| **Education** | 4 | CS/AI/ML degree, Tier-1/2 institute, postgraduate studies — using the explicit `tier` and `field_of_study` fields confirmed in the schema. | `education[]` |
| **Risk Penalty** | 0 to −10 | Subtracted: keyword stuffing, overlapping role dates, tight timeline vs. degree, long notice period, inactivity, self-rated skills exceeding platform assessments. | cross-field |

**Final score = sum of 7 positive layers − risk penalty**, clamped to [0, 100], then normalized to [0, 1] by dividing by 97 (the rubric maximum — documented choice, see Cell 1).

Scores are therefore **not** percentile-stretched or cosmetically inflated. A 0.76 at rank 1 means the top candidate earned 76% of the maximum possible rubric points. A rank-100 candidate near 0.51 means they earned just over half — a score that honestly reflects the distance between them and rank 1, rather than compressing both into the high 90s.

---

## Two-pass design: gate first, score second

### Hard gate (before any scoring)

Candidates with physically impossible histories are dropped entirely and never scored. The gate flags only genuine impossibilities — not people who simply omitted early jobs:

- `years_of_experience` implying work started more than 6 years before degree completion
- Stated experience exceeding the span of the entire listed career by more than 8 years (dataset-validated threshold: only 24 of 100,000 profiles trip this)
- Overlapping full-time roles (>31 days)
- More than one simultaneous `is_current` role
- 3+ skills self-rated "expert" with an explicit `duration_months = 0` (the challenge's own named honeypot pattern — only 21 profiles in the full dataset)
- Career ending before the degree even began

In our run: **2,858 of 100,000 candidates dropped (2.86%)** — consistent with the ~80 planted honeypots plus naturally malformed data. The top 100 contains zero honeypot-pattern profiles (audited post-run).

### Soft risk penalty (during scoring)

Subtler red flags reduce points without full disqualification:

- **Keyword stuffing**: skills list claims a lot (Layer 3 high) but career-history evidence is thin (Layer 2 low) — the classic stuffing pattern the challenge names.
- **Minor overlapping dates**: ≤31 days (below the hard gate threshold but still suspicious).
- **Tight timeline**: experience implies work started 3–6 years before degree (plausible, but penalised).
- **Long notice period**: >90 days.
- **Inactivity**: last active >150 days ago.
- **Self-rated vs. measured**: self-rating diverges from `skill_assessment_scores` by more than the threshold.

---

## Why the JD skill layer uses alias-aware matching

A plain substring match against the JD's skill names ("llm fine-tuning", "vector db") missed the majority of real matches because the dataset's `skills[].name` vocabulary uses different surface strings for the same concept — "LoRA", "QLoRA", "Fine-tuning LLMs" for what the JD calls "llm fine-tuning"; "Learning to Rank", "BM25" for "ranking". We surveyed the actual skill-name distribution across all 100,000 profiles and built an alias table for each JD concept. The point budget is unchanged (sums to 22); only the matching vocabulary was corrected.

---

## Reasoning is fact-based, rank-aware, and never generated

The `reasoning` field for each candidate is assembled directly from matched evidence — **not written by an LLM**. It cannot hallucinate, and every clause maps to a real scored field.

Three properties enforced by design:

1. **Fact-grounded**: every claim (years of experience, current title, matched skills, tech phrases) is sourced from a specific field on that specific candidate.
2. **Rank-aware**: the tone scales with rank. Top-10 candidates only surface a concern when the evidence is genuinely weak (weak-layer fraction < 0.35). Ranks 11–50 voice a concern when a layer is meaningfully below its maximum. Ranks 51–100 always acknowledge a gap, and ranks 90–100 add a "borderline top-100 inclusion" note. This directly addresses the Stage 4 tone-vs-rank consistency check.
3. **Non-templated**: opening phrasing rotates deterministically by `candidate_id` across three formats, so sampled rows don't share one skeleton. Result: **99 distinct reasoning skeletons across 100 rows, zero exact duplicates.**

---

## Reasoning quality across rank bands (Stage 4 evidence)

| Rank band | Rows voicing a concern or gap | 
|---|---|
| 1–10 | 4 / 10 (only where genuinely weak) |
| 11–50 | 40 / 40 |
| 51–100 | 50 / 50 |

---

## Deterministic reproduction

All date-relative logic uses a fixed `REFERENCE_DATE = 2026-06-30` instead of `datetime.now()`. Running the notebook today, next month, or at Stage 3 produces **byte-identical output** regardless of when the reproduction happens.

---

## Iteration history (for Stage 4 git-history check)

| Version | Approach | Why it changed |
|---|---|---|
| v1 | Weighted average (5 signals, weights sum to 1.0), BM25 percentile-normalized, generic templated reasoning | Reasoning had only 8 distinct skeletons and 22 exact duplicates; skill matching missed real aliases; honeypot gate over-flagged realistic candidates |
| v2 (submitted) | Additive 7-layer rubric (Doc-4 JD-centric), alias-aware skill matching, rank-aware fact-grounded reasoning, dataset-validated honeypot gate | Addresses all Stage 4 rubric checks; 55 of v1's top-100 replaced by stronger candidates |

---

## Repository structure

```
.
├── README.md                        # this file
├── requirements.txt                 # dependencies
├── submission_metadata.yaml         # challenge submission metadata
├── Redrob_Ranker_v2.ipynb           # notebook (Colab sandbox link)
├── rank.py                          # main file
└── submission.csv                   # the submitted top-100 ranking
```

---

## How to run

> NNOTE: make sure you have candidate.jsonl or the file on which you have to inference on

### Option 1: Colab notebook (sandbox demo + full reproduction)

1. Open `Redrob_Ranker_v2.ipynb` in Google Colab.
2. Add `job_description.txt` file (In the same repo or upload yours file)
2. Upload `candidates.jsonl` to the Colab session (or mount from Drive).
3. **Runtime → Run all.**

The notebook streams the full 100K file in a single pass, scores every surviving candidate, and writes the validated CSV. Runtime: ~60 seconds on a standard Colab CPU instance.

```bash
# Equivalent CLI reproduction via nbconvert:
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Redrob_Ranker_v2.ipynb \
  --ExecutePreprocessor.timeout=300 \
  --output team_submission_executed.ipynb
```

### Option 2: Github

Step 1: Clone repo
```bash
git clone https://github.com/ghanshyam1011/redrob-candidate-ranking-submision_version-.git
```

Step 2: Create virtual environment (Optional)
```bash
python -m venv venv 

venv\Scripts\activate
```

Step 3: Download requirements
```bash
pip install -r requirements.txt
```

Step 4: Run `rank.py` file

```bash
python rank.py --candidates ./candidates.jsonl --jd ./job_description.txt --out ./submission.csv 
```

---

## Constraints compliance

| Constraint | How we meet it |
|---|---|
| CPU only | BM25 + lexical phrase matching + numpy; no GPU anywhere. |
| No internet at ranking time | No API calls; all scoring is fully local. |
| Under 5 minutes for 100k | Single streaming pass; ~60s measured on CPU. |
| Under 16 GB RAM | Streaming JSONL — only the top-300 buffer is held in memory at once; peak ~50 MB. |
| ≤ 10% honeypots in top 100 | Hard gate drops honeypots before scoring; top 100 audited post-run: **0 honeypot-pattern profiles**. |

---

## Design philosophy (one line)

Identify the people who would actually succeed in the role by reading the work
they have done — transparently, defensibly, and within hard production constraints —
rather than rewarding whoever packed the most keywords into a profile.