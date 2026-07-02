# Redrob Intelligent Candidate Discovery & Ranking

**Team:** AI Quartet &nbsp;|&nbsp; **Event:** India Runs Hackathon / Hack2Skill  
**Members:** Ghanshyam Kumavat · Dhruv Bhrasadiya · Shravani Chavan · Sakshi Patil

---

## What does this project do?

Given **100,000 candidate profiles** and a **Senior AI Engineer job description**, this system finds the **top 100 best-fit candidates** and explains exactly why each one ranked where they did.

It does this without any LLM, GPU, or internet connection at ranking time — just smart, transparent scoring logic that runs in about **60 seconds on a normal laptop CPU**.

---

## Why not just match keywords?

The dataset contains ~80 "honeypot" profiles — fake candidates who stuff their skills list with AI/ML buzzwords but whose actual work history shows no real technical work. A simple keyword-matching system would rank these highly.

Our approach instead asks: **what did this person actually build and ship?**

> Rank by what candidates *did* (career history), not what they *listed* (skills).

---

## How scoring works (simple version)

Every candidate gets points across 7 areas, then a penalty is subtracted for red flags:

| What we check | Max points |
|---|---:|
| Technical depth in career history (retrieval, embeddings, LLMs, etc.) | 28 |
| Match to the JD's specific tools & skills | 22 |
| Evidence of owning and shipping real systems | 15 |
| Platform signals (response rate, GitHub activity, recency) | 12 |
| Career progression (5–9 year sweet spot, title growth) | 10 |
| Company type (AI/product company vs. services firm) | 6 |
| Education (CS/AI degree, Tier-1/2 institute) | 4 |
| **Risk penalty** (keyword stuffing, fake timelines, etc.) | −10 |

**Final score = sum of 7 layers − penalty**, then divided by 97 (the max possible) to get a 0–1 number.

A score of **0.76 means the candidate earned 76% of the maximum rubric points** — no artificial inflation.

---

## Repository structure

```
redrob/
├── README.md                    ← you are here
├── requirements.txt             ← Python packages needed
├── submission_metadata.yaml     ← hackathon submission info
├── job_description.txt          ← the JD we ranked against
├── rank.py                    ← main ranking script (run this)
├── notebooks/
│   ├── Redrob_Ranker_v2.ipynb   ← Colab notebook (sandbox demo)
│   └── redrob_eda.ipynb         ← exploratory analysis (optional)
├── streamlit_ui/
│   └── app.py                   ← HuggingFace Spaces demo
└── output/
    └── team_ai_quartet.csv      ← our final submitted top-100
```

> **Note:** `candidates.jsonl` (487 MB) is NOT in this repo — it is too large for GitHub.
> Download it from the hackathon portal and place it in the root folder before running.

---

## Quickstart — run it yourself

### Step 1: Clone the repo

```bash
git clone https://github.com/ghanshyam1011/redrob-candidate-ranking-submision_version-.git
cd redrob-candidate-ranking-submision_version-
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add the candidates file

Place `candidates.jsonl` (or `candidates.jsonl.gz`) in the root folder.  
The script auto-detects which one is present — `.gz` is preferred (much smaller to upload).

### Step 4: Run

```bash
python rank.py --jd job_description.txt --out output/my_submission.csv
```

That's it. The script will print progress and write the ranked CSV in ~60 seconds.

**Optional — specify the candidates file path explicitly:**
```bash
python rank.py --candidates candidates.jsonl.gz --jd job_description.txt --out output/my_submission.csv
```

---

## Run in Google Colab (no local setup needed)

1. Open [`Redrob_Ranker_v2.ipynb`](notebooks/Redrob_Ranker_v2.ipynb) in [Google Colab](https://colab.research.google.com)
2. Upload `candidates.jsonl` or `candidates.jsonl.gz` to the Colab session
3. Click **Runtime → Run all**
4. The ranked CSV downloads automatically when done

> **Tip:** Upload `candidates.jsonl.gz` instead of the raw `.jsonl` — it is ~6x smaller so it uploads much faster. The notebook handles both formats automatically.

**Live demo (Hugging Face):** [https://huggingface.co/spaces/LegalAIShravani/redrob-ranker](https://huggingface.co/spaces/LegalAIShravani/redrob-ranker)

---

## Output format

The output CSV has exactly 4 columns:

| Column | Example |
|---|---|
| `candidate_id` | `CAND_0008425` |
| `rank` | `1` |
| `score` | `0.7629` |
| `reasoning` | `7.8y experience, currently Senior NLP Engineer; hands-on retrieval/search evidence; built production systems, owned architecture; JD-aligned skills: python, retrieval, pgvector` |

The `reasoning` field is assembled from actual scored evidence — never written by an AI — so every sentence maps to a real data point on that candidate.

---

## How we handle honeypots

Before any scoring, a **hard gate** drops candidates with physically impossible histories:

- Experience that implies working 6+ years before finishing a degree
- Stated experience far exceeding the span of their entire listed career
- Overlapping full-time jobs
- Multiple simultaneous "current" roles
- 3+ skills marked "expert" with explicitly 0 months of use

**Result:** 2,858 of 100,000 profiles dropped before scoring. Zero honeypot-pattern profiles in the final top 100.

---

## Performance

| Metric | Value |
|---|---|
| Runtime | ~60 seconds for 100,000 candidates |
| Peak RAM | ~50 MB (streams the file, never loads everything into memory) |
| GPU needed | No |
| Internet at ranking time | No |
| Honeypots in top 100 | 0 |

---

## How we iterated (v1 → v2)

| | v1 (baseline) | v2 (submitted) |
|---|---|---|
| Scoring | Weighted average, 5 signals | Additive 7-layer rubric |
| Skill matching | Plain substring | Alias-aware (LoRA → llm fine-tuning) |
| Reasoning | 8 templates, 22 duplicates | 99 distinct skeletons, 0 duplicates |
| Honeypots in top 100 | 2 | 0 |
| Deterministic output | No (used `datetime.now()`) | Yes (fixed `REFERENCE_DATE`) |

---

## Requirements

```
rank-bm25
pandas
```

No GPU, no heavy ML libraries, no API keys needed.

---
