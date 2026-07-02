import argparse
import csv
import gzip
import heapq
import json
import re
import time
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
REFERENCE_DATE = datetime(2026, 6, 30)
REFERENCE_DATE_STR = "2026-06-30"

CONFIG = {
    "layer_points": {
        "technical_substance": 28,
        "jd_skill_alignment": 22,
        "production_engineering": 15,
        "behavioral_hireability": 12,
        "career_progression": 10,
        "company_context": 6,
        "education": 4,
    },
    "risk_penalty_max": 10,
    "technical_substance_points": {
        "retrieval_search": 7, "embeddings": 5, "vector_databases": 4,
        "ranking_reranking": 4, "llm_finetuning": 3, "eval_frameworks": 3,
        "distributed_ml_inference": 2,
    },
    "production_engineering_points": {
        "built_production_system": 4, "owned_architecture": 3, "deployed_to_users": 3,
        "current_hands_on_engineer": 2, "scaling_optimization": 2, "monitoring_evaluation": 1,
    },
    "behavioral_points": {
        "open_to_work": 2, "recruiter_response_rate": 2, "interview_completion_rate": 2,
        "github_activity": 2, "recently_active": 1, "search_appearances": 1,
        "saved_by_recruiters": 1, "profile_completeness": 1,
    },
    "career_progression_points": {
        "years_5_to_9": 3, "promotions_growth": 2, "stable_tenure": 2,
        "recent_engineering_role": 2, "leadership_mentoring": 1,
    },
    "company_context_points": {
        "ai_product_company": 3, "product_engineering": 2, "large_scale_systems_exposure": 1,
    },
    "education_points": {
        "cs_ai_ml_degree": 2, "tier_1_2_institute": 1, "relevant_higher_studies": 1,
    },
    "risk_penalty_points": {
        "keyword_stuffing": -3, "contradictory_career_history": -3,
        "impossible_timeline_soft": -2, "extremely_long_notice_period": -1,
        "no_recent_activity": -1, "self_rated_skill_exceeds_measured": -2,
    },
    "consistency_penalty": {"enabled": True, "gap_threshold": 20, "severe_gap_threshold": 35},
    "honeypot_rules": {
        "check_overlapping_roles": True,
        "max_allowed_simultaneous_current_roles": 1,
        "max_exp_minus_span_years": 8,
        "min_expert_zero_month_skills": 3,
    },
    "proficiency_scale": {"beginner": 25, "intermediate": 55, "advanced": 85},
    "top_n": 100,
    "heap_buffer": 300,
    "output_columns": ["candidate_id", "rank", "score", "reasoning"],
}
_RUBRIC_MAX = sum(CONFIG["layer_points"].values())  # 97
assert _RUBRIC_MAX == 97
assert CONFIG["output_columns"] == ["candidate_id", "rank", "score", "reasoning"]

DATE_FMT = "%Y-%m-%d"


# ============================================================
# DATA LOADING
# ============================================================
def _open_path(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")


def load_jd(path):
    with _open_path(path) as f:
        text = f.read()
    base = (path[:-3] if path.endswith(".gz") else path).lower()
    if base.endswith(".json"):
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = parsed.get("description") or parsed.get("job_description") or text
    return text


# ============================================================
# HONEYPOT GATE (6 checks, hard disqualify before scoring)
# ============================================================
def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, DATE_FMT)
    except (ValueError, TypeError):
        return None


def run_honeypot_gate(cand, config):
    rules = config["honeypot_rules"]
    reasons = []
    profile = cand.get("profile") or {}
    history = cand.get("career_history") or []
    education = cand.get("education") or []
    yexp = profile.get("years_of_experience")

    if yexp is not None:
        if yexp > 60:
            return False, ["experience exceeds human maximum"]
        deg_ends = [e.get("end_year") for e in education if e.get("end_year")]
        if deg_ends and (REFERENCE_DATE.year - yexp) < min(deg_ends) - 6:
            reasons.append("experience implies working before degree completion")
        starts = [_parse_date(h.get("start_date")) for h in history]
        starts = [d for d in starts if d]
        if starts:
            span = (REFERENCE_DATE - min(starts)).days / 365.25
            if yexp > span + rules["max_exp_minus_span_years"]:
                reasons.append(f"claims {yexp}y but career spans {span:.1f}y")

    if rules["check_overlapping_roles"]:
        intervals = [(s, e) for h in history if not h.get("is_current")
                     for s, e in [(_parse_date(h.get("start_date")), _parse_date(h.get("end_date")))]
                     if s and e]
        intervals.sort()
        for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
            if (e1 - s2).days > 31:
                reasons.append("overlapping full-time roles"); break

    cur = sum(1 for h in history if h.get("is_current"))
    if cur > rules["max_allowed_simultaneous_current_roles"]:
        reasons.append(f"{cur} simultaneous current roles")

    deg_starts = [e.get("start_year") for e in education if e.get("start_year")]
    if deg_starts and history:
        ends = [REFERENCE_DATE if h.get("is_current") else _parse_date(h.get("end_date")) for h in history]
        ends = [e for e in ends if e]
        if ends and max(ends).year < min(deg_starts) - 1:
            reasons.append("career ends before degree begins")

    zero_expert = sum(1 for s in cand.get("skills") or []
                      if s.get("proficiency") == "expert" and s.get("duration_months") == 0)
    if zero_expert >= rules["min_expert_zero_month_skills"]:
        reasons.append(f"{zero_expert} expert skills with 0 months used")

    return len(reasons) == 0, reasons


# ============================================================
# LAYER PHRASE TABLES
# ============================================================
TECHNICAL_SUBSTANCE_PHRASES = {
    "retrieval_search": ["retrieval", "search system", "search engine", "information retrieval",
                         "semantic search", "query understanding", "recommendation system",
                         "document retrieval", "search infrastructure", "search relevance", "indexing"],
    "embeddings": ["embedding", "embeddings", "vector representation", "sentence embedding",
                   "dense retrieval", "text embedding", "embedding model", "encoder model"],
    "vector_databases": ["vector database", "milvus", "faiss", "pinecone", "qdrant", "weaviate",
                         "vector store", "vector index", "ann search", "pgvector"],
    "ranking_reranking": ["ranking", "re-ranking", "reranking", "learning to rank",
                          "relevance ranking", "rank model", "scoring model", "ranking pipeline"],
    "llm_finetuning": ["fine-tuning", "fine tuned", "finetune", "lora", "peft",
                       "instruction tuning", "rlhf", "llm", "large language model", "model tuning"],
    "eval_frameworks": ["ndcg", "map@", "mrr", "precision@", "recall@", "evaluation framework",
                        "offline evaluation", "a/b test", "eval pipeline", "model evaluation"],
    "distributed_ml_inference": ["distributed training", "model inference", "inference pipeline",
                                 "model serving", "low latency", "high throughput", "scalable inference"],
}
_SATURATION_HITS = 2

OWNERSHIP_VERBS = ["owned", "own ", "led", "leading", "built and shipped", "shipped", "deployed",
                   "architected", "designed and built", "drove", "spearheaded", "on-call", "on call",
                   "end to end", "end-to-end", "from scratch", "took ownership", "responsible for", "scaled"]
HEDGE_WORDS = ["exposure", "exposed to", "assisted", "supported the", "helped the", "familiar with",
               "worked closely with", "adjacent", "some exposure", "involved in", "contributed to",
               "learning", "building competence", "interested in", "transitioning"]

PRODUCTION_PHRASES = {
    "built_production_system": ["built", "shipped", "built and shipped", "developed and launched"],
    "owned_architecture": ["owned", "architected", "designed and built", "took ownership", "system design"],
    "deployed_to_users": ["deployed", "launched", "released to production", "in production", "live in production"],
    "scaling_optimization": ["scaled", "scaling", "optimized", "latency reduction", "throughput", "cost reduction"],
    "monitoring_evaluation": ["on-call", "on call", "monitoring", "observability", "incident response", "reliability"],
}

_JD_SKILL_STANDALONE = {
    "python":             (["python"], 3),
    "retrieval":          (["retrieval", "information retrieval", "semantic search", "rag", "dense retrieval"], 3),
    "vector_db":          (["vector search", "pgvector", "pinecone", "qdrant", "faiss", "milvus", "weaviate", "vector database"], 3),
    "llm_finetuning":     (["fine-tuning llms", "lora", "qlora", "peft", "instruction tuning", "rlhf", "fine-tuning", "finetuning"], 2),
    "ranking":            (["ranking", "learning to rank", "bm25", "recommendation systems", "reranking"], 2),
    "evaluation_metrics": (["ndcg", "mrr", "map@", "precision@", "recall@", "evaluation metrics"], 2),
    "spark":              (["spark", "pyspark"], 1),
    "airflow":            (["airflow"], 1),
    "sql":                (["sql", "postgresql", "snowflake", "dbt"], 1),
}
_JD_SKILL_GROUPS = {
    "vector_db_tools": (["milvus", "pinecone", "qdrant", "faiss", "weaviate", "pgvector"], 2),
    "container_tools": (["docker", "kubernetes", "k8s"], 1),
    "github_oss":      (["github", "open source", "open-source"], 1),
}
_KEY_JD_SKILLS = {"python", "retrieval", "vector_db", "llm_finetuning", "ranking", "evaluation_metrics"}

SENIORITY_LEVELS = [
    (6, ["chief", "cto", "vp ", "vice president", "head of", "director"]),
    (5, ["principal", "staff", "distinguished"]),
    (4, ["lead", "manager", "founding", "founder"]),
    (3, ["senior", "sr.", "sr "]),
    (2, ["engineer", "developer", "scientist", "analyst", "specialist"]),
    (1, ["junior", "jr.", "jr ", "intern", "trainee", "associate", "graduate"]),
]
_TECH_TITLE_TERMS = ["engineer", "developer", "scientist", "researcher", "architect",
                     "ml", "ai", "data", "software", "backend", "infrastructure"]
_LEADERSHIP_TERMS = ["mentored", "mentor", "led a team", "managed engineers", "team lead",
                     "manager", "led the team", "coached", "supervised"]

_SERVICES_IND = {"IT Services", "Consulting"}
_SERVICES_CO = {"tcs", "infosys", "wipro", "accenture", "cognizant",
                "capgemini", "tech mahindra", "hcl", "ltimindtree"}
_AI_PRODUCT_INDUSTRIES = {"Artificial Intelligence", "AI", "Machine Learning",
                          "Technology", "Software", "Internet", "SaaS", "AI/ML"}
_MID_LARGE = {"201-500", "501-1000", "1001-5000", "5001-10000", "10001+"}
_SMALL_STARTUP = {"11-50", "51-200"}
_LARGE = {"1001-5000", "5001-10000", "10001+"}

_CS_AI_ML_FIELDS = {"Computer Science", "Computer Engineering", "Artificial Intelligence",
                    "Machine Learning", "Data Science", "Information Technology"}
_ENG_ADJACENT = {"Electronics", "Electrical Engineering", "Mathematics", "Statistics", "Physics"}


# ============================================================
# SCORING FUNCTIONS
# ============================================================
def _days_since(date_str):
    try:
        return (REFERENCE_DATE - datetime.strptime(date_str, DATE_FMT)).days
    except (TypeError, ValueError):
        return None


def _title_seniority(title):
    tl = (title or "").lower()
    for level, kws in SENIORITY_LEVELS:
        if any(k in tl for k in kws):
            return level
    return 2


def _start_key(role):
    return _parse_date(role.get("start_date")) or datetime.min


def score_technical_substance(tech_text, config):
    breakdown, total = {}, 0.0
    for cat, max_pts in config["technical_substance_points"].items():
        hits = [p for p in TECHNICAL_SUBSTANCE_PHRASES[cat] if p in tech_text]
        earned = round(max_pts * min(1.0, len(hits) / _SATURATION_HITS), 2)
        total += earned
        breakdown[cat] = {"earned": earned, "hits": hits}
    return min(total, config["layer_points"]["technical_substance"]), breakdown


def score_jd_skill_alignment(skills_text, config):
    matched, missing_key, total = {}, [], 0.0
    for key, (aliases, pts) in _JD_SKILL_STANDALONE.items():
        hit = next((a for a in aliases if a in skills_text), None)
        if hit:
            matched[hit] = pts; total += pts
        elif key in _KEY_JD_SKILLS:
            missing_key.append(key)
    for _, (terms, pts) in _JD_SKILL_GROUPS.items():
        hit = next((t for t in terms if t in skills_text), None)
        if hit:
            if hit not in matched: matched[hit] = pts
            total += pts
    return min(total, config["layer_points"]["jd_skill_alignment"]), matched, missing_key


def score_production_engineering(history, desc_text, config):
    pts_table = config["production_engineering_points"]
    hedged = any(h in desc_text for h in HEDGE_WORDS)
    breakdown, total = {}, 0.0
    for cat, max_pts in pts_table.items():
        if cat == "current_hands_on_engineer": continue
        if any(p in desc_text for p in PRODUCTION_PHRASES[cat]):
            earned = max_pts * (0.5 if hedged else 1.0)
            total += earned; breakdown[cat] = {"earned": round(earned, 2), "hedged": hedged}
        else:
            breakdown[cat] = {"earned": 0.0}
    max_hands = pts_table["current_hands_on_engineer"]
    earned = 0.0
    current = [r for r in history if r.get("is_current")]
    if current:
        title = (current[0].get("title") or "").lower()
        desc = (current[0].get("description") or "").lower()
        if not any(k in title for k in ["director", "vp ", "vice president", "head of"]):
            earned = max_hands if any(v in desc for v in OWNERSHIP_VERBS) else max_hands * 0.5
    total += earned; breakdown["current_hands_on_engineer"] = {"earned": round(earned, 2)}
    return min(total, config["layer_points"]["production_engineering"]), breakdown


def score_behavioral_hireability(sig, config):
    pt, total = config["behavioral_points"], 0.0

    def _rate(v):
        if isinstance(v, (int, float)):
            return v if v <= 1.0 else min(1.0, v / 100.0)
        return None

    v = sig.get("open_to_work_flag")
    total += pt["open_to_work"] * (0.5 if v is None else (1.0 if v else 0.0))
    r = _rate(sig.get("recruiter_response_rate"))
    total += pt["recruiter_response_rate"] * (r or 0.0)
    r = _rate(sig.get("interview_completion_rate"))
    total += pt["interview_completion_rate"] * (r or 0.0)
    gh = sig.get("github_activity_score")
    frac = (gh if isinstance(gh, (int, float)) and gh <= 1.0 else min(1.0, gh / 100.0)) if isinstance(gh, (int, float)) else 0.5
    total += pt["github_activity"] * frac
    days = _days_since(sig.get("last_active_date"))
    total += pt["recently_active"] * (1.0 if days is not None and days <= 30 else 0.5 if days is not None and days <= 90 else 0.0)
    sa = sig.get("search_appearance_30d")
    total += pt["search_appearances"] * (min(1.0, float(sa) / 105.0) if isinstance(sa, (int, float)) else 0.5)
    sv = sig.get("saved_by_recruiters_30d")
    total += pt["saved_by_recruiters"] * (min(1.0, float(sv) / 5.0) if isinstance(sv, (int, float)) else 0.5)
    pc = sig.get("profile_completeness_score")
    total += pt["profile_completeness"] * (min(1.0, pc / 100.0) if isinstance(pc, (int, float)) else 0.5)
    return min(total, config["layer_points"]["behavioral_hireability"])


def score_career_progression(profile, history, config):
    pt, total = config["career_progression_points"], 0.0
    years = profile.get("years_of_experience")
    earned = 0.0
    if isinstance(years, (int, float)):
        if 5 <= years <= 9: earned = pt["years_5_to_9"]
        elif 3 <= years < 5: earned = pt["years_5_to_9"] * (years - 3) / 2.0
        elif 9 < years <= 12: earned = pt["years_5_to_9"] * (12 - years) / 3.0
    total += earned
    ordered = sorted(history, key=_start_key)
    net = (_title_seniority(ordered[-1].get("title")) - _title_seniority(ordered[0].get("title"))) if ordered else 0
    total += pt["promotions_growth"] * (1.0 if net >= 2 else 0.75 if net == 1 else 0.25 if net == 0 else 0.0)
    past = [r for r in history if not r.get("is_current")]
    avg = (sum(r.get("duration_months") or 0 for r in past) / len(past)) if past else 0
    total += pt["stable_tenure"] * (1.0 if avg >= 18 else 0.5 if avg >= 12 else 0.0)
    cur = [r for r in history if r.get("is_current")]
    ref = cur[0] if cur else (ordered[-1] if ordered else None)
    if ref and any(t in (ref.get("title") or "").lower() for t in _TECH_TITLE_TERMS):
        total += pt["recent_engineering_role"]
    full_text = " ".join((r.get("title") or "") + " " + (r.get("description") or "") for r in history).lower()
    if any(t in full_text for t in _LEADERSHIP_TERMS):
        total += pt["leadership_mentoring"]
    return min(total, config["layer_points"]["career_progression"])


def score_company_context(profile, history, config):
    pt, total = config["company_context_points"], 0.0
    cur = [r for r in history if r.get("is_current")]
    role = cur[0] if cur else (sorted(history, key=_start_key)[-1] if history else None)
    if not role: return 0.0, False
    industry = role.get("industry") or ""
    company = (role.get("company") or "").lower()
    desc = (role.get("description") or "").lower()
    is_svc = (industry in _SERVICES_IND) or any(s in company for s in _SERVICES_CO)
    total += 0.0 if is_svc else (pt["ai_product_company"] if industry in _AI_PRODUCT_INDUSTRIES else pt["ai_product_company"] * 0.5)
    size = profile.get("current_company_size") or role.get("company_size") or ""
    if not is_svc:
        total += (pt["product_engineering"] if size in _MID_LARGE else
                  pt["product_engineering"] * 0.75 if size in _SMALL_STARTUP else
                  pt["product_engineering"] * 0.5 if "product" in desc else pt["product_engineering"] * 0.25)
    total += pt["large_scale_systems_exposure"] if (size in _LARGE or any(t in desc for t in ["scale", "millions of users", "large-scale"])) else 0.0
    return min(total, config["layer_points"]["company_context"]), is_svc


def score_education(education, config):
    pt, total = config["education_points"], 0.0
    if not education: return 0.0
    fields = {e.get("field_of_study", "") for e in education}
    total += pt["cs_ai_ml_degree"] if fields & _CS_AI_ML_FIELDS else (pt["cs_ai_ml_degree"] * 0.5 if fields & _ENG_ADJACENT else 0.0)
    tiers = {e.get("tier", "") for e in education}
    total += pt["tier_1_2_institute"] if "tier_1" in tiers else (pt["tier_1_2_institute"] * 0.75 if "tier_2" in tiers else 0.0)
    degrees = " ".join(str(e.get("degree", "")).lower() for e in education)
    if len(education) >= 2 or any(t in degrees for t in ["master", "m.tech", "m.s.", "phd", "mba"]):
        total += pt["relevant_higher_studies"]
    return min(total, config["layer_points"]["education"])


def score_risk_penalty(cand, tech_score, skill_score, config):
    rp, items, total = config["risk_penalty_points"], {}, 0.0
    # keyword stuffing
    if (skill_score / config["layer_points"]["jd_skill_alignment"] >= 0.5 and
            tech_score / config["layer_points"]["technical_substance"] <= 0.2):
        items["keyword_stuffing"] = rp["keyword_stuffing"]; total += rp["keyword_stuffing"]
    # soft overlap
    intervals = [(s, e) for h in cand.get("career_history") or [] if not h.get("is_current")
                 for s, e in [(_parse_date(h.get("start_date")), _parse_date(h.get("end_date")))] if s and e]
    intervals.sort()
    for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
        if 0 < (e1 - s2).days <= 31:
            items["contradictory_career_history"] = rp["contradictory_career_history"]
            total += rp["contradictory_career_history"]; break
    # soft timeline
    yexp = (cand.get("profile") or {}).get("years_of_experience")
    ends = [e.get("end_year") for e in cand.get("education") or [] if e.get("end_year")]
    if yexp and ends:
        implied = REFERENCE_DATE.year - yexp
        if min(ends) - 6 <= implied < min(ends) - 3:
            items["impossible_timeline_soft"] = rp["impossible_timeline_soft"]
            total += rp["impossible_timeline_soft"]
    # notice
    notice = (cand.get("redrob_signals") or {}).get("notice_period_days")
    if isinstance(notice, (int, float)) and notice > 90:
        items["extremely_long_notice_period"] = rp["extremely_long_notice_period"]
        total += rp["extremely_long_notice_period"]
    # inactivity
    days = _days_since((cand.get("redrob_signals") or {}).get("last_active_date"))
    if days is not None and days > 150:
        items["no_recent_activity"] = rp["no_recent_activity"]; total += rp["no_recent_activity"]
    # consistency
    cfg = config["consistency_penalty"]
    if cfg["enabled"]:
        scale = config["proficiency_scale"]
        measured = (cand.get("redrob_signals") or {}).get("skill_assessment_scores") or {}
        worst = max((scale[s.get("proficiency", "").lower()] - measured[s["name"]]
                     for s in cand.get("skills") or []
                     if s.get("name") in measured and s.get("proficiency", "").lower() in scale), default=0)
        if worst >= cfg["severe_gap_threshold"]:
            items["self_rated_skill_exceeds_measured"] = rp["self_rated_skill_exceeds_measured"]
            total += rp["self_rated_skill_exceeds_measured"]
        elif worst >= cfg["gap_threshold"]:
            total += rp["self_rated_skill_exceeds_measured"] * 0.5
    return max(total, -config["risk_penalty_max"]), items


# ============================================================
# REASONING
# ============================================================
_TECH_LABEL = {
    "retrieval_search": "retrieval/search", "embeddings": "embeddings",
    "vector_databases": "vector databases", "ranking_reranking": "ranking",
    "llm_finetuning": "LLM fine-tuning", "eval_frameworks": "evaluation metrics",
    "distributed_ml_inference": "distributed ML/inference",
}
_PROD_LABEL = {
    "built_production_system": "built production systems", "owned_architecture": "owned architecture",
    "deployed_to_users": "deployed to production", "scaling_optimization": "scaled/optimized systems",
    "monitoring_evaluation": "on-call/monitoring ownership", "current_hands_on_engineer": "currently hands-on IC",
}
_FLAG_LABEL = {
    "keyword_stuffing": "skills list not well corroborated by career history",
    "contradictory_career_history": "minor overlapping role dates",
    "impossible_timeline_soft": "tight timeline vs degree completion",
    "extremely_long_notice_period": "long notice period",
    "no_recent_activity": "inactive recently",
    "self_rated_skill_exceeds_measured": "self-rated skills exceed measured assessment",
}
_KEY_JD_SKILL_LABELS = {
    "python": "Python", "retrieval": "retrieval", "vector_db": "vector DB tools",
    "llm_finetuning": "LLM fine-tuning", "ranking": "ranking", "evaluation_metrics": "evaluation metrics",
}
_ALL_LAYERS = ["technical_substance", "jd_skill_alignment", "production_engineering",
               "behavioral_hireability", "career_progression", "company_context", "education"]


def _opener(cid, years, title):
    v = int(cid[-2:]) % 3 if cid and cid[-2:].isdigit() else 0
    if isinstance(years, (int, float)) and title:
        return [f"{years:.1f}y experience, currently {title}",
                f"{title} with {years:.1f}y of experience",
                f"Currently {title} ({years:.1f}y experience)"][v]
    return f"Currently {title}" if title else (f"{years:.1f}y experience" if isinstance(years, (int, float)) else "")


def build_reasoning(cand, subs, penalty_items, rank=None):
    profile = cand.get("profile") or {}
    cid = cand.get("candidate_id", "")
    lp = CONFIG["layer_points"]
    parts = [_opener(cid, profile.get("years_of_experience"), profile.get("current_title") or "")]

    tech_bd = subs["tech_bd"]
    tech_hits = sorted(((v["earned"], c, v["hits"]) for c, v in tech_bd.items() if v.get("hits")), key=lambda x: -x[0])
    if tech_hits:
        labels = " and ".join(_TECH_LABEL[c] for _, c, _ in tech_hits[:2])
        parts.append(f'hands-on {labels} evidence (career history mentions "{tech_hits[0][2][0]}")')

    prod_hits = [k for k, v in subs["prod_bd"].items() if v.get("earned", 0) > 0]
    if prod_hits:
        parts.append(", ".join(_PROD_LABEL[k] for k in prod_hits[:3]))

    if subs["matched"]:
        parts.append("JD-aligned skills: " + ", ".join(s.replace("_", " ") for s in list(subs["matched"])[:4]))

    if subs["edu"] / lp["education"] >= 0.75:
        parts.append("Tier-1/2 institute background" if subs.get("has_tier12") else "strong CS/AI/ML education background")

    if subs["company"] / lp["company_context"] >= 0.75:
        parts.append("AI/product company background")

    # weak-layer concern
    fracs = [(k, subs["scores"][k] / lp[k]) for k in _ALL_LAYERS]
    weak_k, weak_f = min(fracs, key=lambda x: x[1])

    flag_part = ("risk flag: " + "; ".join(_FLAG_LABEL[k] for k in penalty_items)) if penalty_items else ""
    if flag_part:
        parts.append(flag_part)

    # rank-aware concern (only once ranks are known)
    if rank is not None:
        concern = _concern(cand, subs, weak_k, weak_f)
        if concern:
            if rank <= 10 and weak_f < 0.35:
                parts.append(f"minor concern: {concern}")
            elif rank <= 50 and weak_f < 0.6:
                parts.append(f"concern: {concern}")
            elif rank > 50:
                parts.append(f"gap: {concern}")
        if rank >= 90:
            parts.append("borderline top-100 inclusion")

    r = "; ".join(p for p in parts if p)
    return (r[0].upper() + r[1:]) if r else "Scored on available signals."


def _concern(cand, subs, weak_k, weak_f):
    sig = cand.get("redrob_signals") or {}
    profile = cand.get("profile") or {}
    if weak_k == "jd_skill_alignment":
        missing = subs.get("missing_key") or []
        if missing:
            return "JD skills not evidenced: " + ", ".join(_KEY_JD_SKILL_LABELS[k] for k in missing[:3])
        return "partial JD skill coverage"
    if weak_k == "technical_substance":
        return "thin hands-on retrieval/embedding evidence in career history"
    if weak_k == "production_engineering":
        if any(v.get("hedged") for v in subs["prod_bd"].values() if isinstance(v, dict)):
            return "production claims use hedged language, suggesting support work rather than full ownership"
        return "little production ownership or deployment evidence"
    if weak_k == "behavioral_hireability":
        rr = sig.get("recruiter_response_rate")
        return f"low recruiter response rate ({rr:.2f})" if isinstance(rr, (int, float)) and rr <= 0.35 else "weak platform engagement signals"
    if weak_k == "career_progression":
        y = profile.get("years_of_experience")
        return f"{y:.1f}y experience outside the JD's 5-9y target band" if isinstance(y, (int, float)) and not (5 <= y <= 9) else "flat title progression"
    if weak_k == "company_context":
        return "current role is at a services firm, not a product company" if subs.get("is_svc") else "no confirmed AI/product company background"
    if weak_k == "education":
        return "no Tier-1/2 institute or CS/AI/ML degree listed"
    return ""


# ============================================================
# MASTER SCORER
# ============================================================
def score_candidate(cand, config):
    profile = cand.get("profile") or {}
    history = cand.get("career_history") or []
    sig = cand.get("redrob_signals") or {}
    education = cand.get("education") or []

    summary = profile.get("summary") or ""
    role_text = " ".join((r.get("title") or "") + ". " + (r.get("description") or "") for r in history)
    tech_text = (summary + " " + role_text).lower()
    desc_text = " ".join(r.get("description") or "" for r in history).lower()
    skills_text = (" | ".join(s.get("name") or "" for s in cand.get("skills") or []) + " | " + summary).lower()

    tech, tech_bd = score_technical_substance(tech_text, config)
    skill, matched, missing_key = score_jd_skill_alignment(skills_text, config)
    prod, prod_bd = score_production_engineering(history, desc_text, config)
    behav = score_behavioral_hireability(sig, config)
    career = score_career_progression(profile, history, config)
    company, is_svc = score_company_context(profile, history, config)
    edu = score_education(education, config)
    penalty, penalty_items = score_risk_penalty(cand, tech, skill, config)

    # check tier for reasoning
    tiers = {e.get("tier", "") for e in education}
    has_tier12 = "tier_1" in tiers or "tier_2" in tiers

    lp = config["layer_points"]
    scores = {"technical_substance": tech, "jd_skill_alignment": skill,
              "production_engineering": prod, "behavioral_hireability": behav,
              "career_progression": career, "company_context": company, "education": edu}
    raw = sum(scores.values()) + penalty
    final = max(0.0, min(100.0, raw))

    subs = {"scores": scores, "tech_bd": tech_bd, "prod_bd": prod_bd,
            "matched": matched, "missing_key": missing_key,
            "edu": edu, "company": company, "has_tier12": has_tier12, "is_svc": is_svc}

    return {
        "candidate_id": cand.get("candidate_id"),
        "score": round(final, 2),
        "penalty_items": penalty_items,
        "subs": subs,
        "_cand": cand,
    }


# ============================================================
# PIPELINE (streaming)
# ============================================================
def run_pipeline(path, config):
    t0 = time.time()
    heap, seq, n_total, n_disq, n_bad = [], 0, 0, 0, 0
    buf = config["heap_buffer"]

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for raw in f:
            if not raw.strip(): continue
            try:
                cand = json.loads(raw)
            except Exception:
                n_bad += 1; continue
            n_total += 1
            if not run_honeypot_gate(cand, config)[0]:
                n_disq += 1; continue
            res = score_candidate(cand, config)
            seq += 1
            item = (res["score"], -seq, res)
            if len(heap) < buf:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    results = [it[2] for it in heap]
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    top = results[:config["top_n"]]

    final_rows = []
    for i, r in enumerate(top, 1):
        reasoning = build_reasoning(r["_cand"], r["subs"], r["penalty_items"], rank=i)
        final_rows.append({
            "candidate_id": r["candidate_id"],
            "rank": i,
            "score": round(r["score"] / _RUBRIC_MAX, 4),
            "reasoning": reasoning,
        })

    elapsed = time.time() - t0
    print(f"  {n_total:,} parsed, {n_disq:,} dropped ({100*n_disq/n_total:.1f}%), "
          f"{len(final_rows)} in top-{config['top_n']} | {elapsed:.1f}s | "
          f"{n_total/elapsed:.0f} candidates/s")
    assert elapsed < 300, "exceeded 5-minute budget"
    return final_rows


# ============================================================
# OUTPUT
# ============================================================
def validate_and_write(rows, path, config):
    n = config["top_n"]
    assert len(rows) == n
    assert [r["rank"] for r in rows] == list(range(1, n + 1))
    for i in range(1, len(rows)):
        assert rows[i]["score"] <= rows[i - 1]["score"], "scores not non-increasing"
    assert all(r["reasoning"].strip() for r in rows)
    assert len({r["candidate_id"] for r in rows}) == n

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=config["output_columns"], extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"  wrote {len(rows)} rows to {path} (validated)")


def main():
    ap = argparse.ArgumentParser(description="Redrob candidate ranker (v2 additive)")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd", required=True)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    print("Ranking...")
    top = run_pipeline(args.candidates, CONFIG)
    print("Writing...")
    validate_and_write(top, args.out, CONFIG)
    print("Done.")


if __name__ == "__main__":
    main()
