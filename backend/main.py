from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from ortools.sat.python import cp_model
import math
import os
import json

app = FastAPI()

raw = os.getenv("_ALLOWED_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or ""
try:
    parsed = json.loads(raw) if raw.strip().startswith("[") else None
except json.JSONDecodeError:
    parsed = None

origins = (
    [s.strip() for s in raw.split(",") if s.strip()]
    if not parsed else
    [str(s).strip() for s in parsed]
)
if not origins:
    origins = ["http://localhost:3000"]  # sensible default for local dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mentor(BaseModel):
    id: str
    name: Optional[str] = None
    major: str
    extracurriculars: List[str] = Field(default_factory=list)
    capacity: int = 1

class Mentee(BaseModel):
    id: str
    name: Optional[str] = None
    target_major: str
    extracurriculars: List[str] = Field(default_factory=list)

class Scoring(BaseModel):
    priority_mode: Literal["lexicographic","weighted"] = "weighted"
    bigM: int = 1000
    extras_metric: Literal["count","jaccard","cosine"] = "count"
    major_rule: Literal["exact"] = "exact"  # room to extend later

class SolveRequest(BaseModel):
    mentors: List[Mentor]
    mentees: List[Mentee]
    scoring: Scoring

@app.get("/health")
def health():
    return {"ok": True}

def extras_score(a: List[str], b: List[str], metric: str) -> float:
    A, B = set(map(str.lower, a)), set(map(str.lower, b))
    if metric == "count":
        return float(len(A & B))
    if metric == "jaccard":
        return float(len(A & B)) / (len(A | B) or 1)
    # cosine on multi-hot over union vocab
    vocab = sorted(A | B)
    va = [1.0 if x in A else 0.0 for x in vocab]
    vb = [1.0 if x in B else 0.0 for x in vocab]
    dot = sum(x*y for x, y in zip(va, vb))
    na = math.sqrt(sum(x*x for x in va))
    nb = math.sqrt(sum(y*y for y in vb))
    return dot / (na*nb or 1.0)

def major_match(mentor_major: str, target_major: str) -> int:
    return 1 if mentor_major.strip().lower() == target_major.strip().lower() else 0

def build_model(major, extras, caps):
    """Return (model, x, major_sum, extras_sum) for fresh solve passes."""
    I = range(len(caps))
    J = range(len(extras[0]) if extras else 0)
    m = cp_model.CpModel()
    x = {(i, j): m.NewBoolVar(f"x_{i}_{j}") for i in I for j in J}

    # Each mentee exactly one mentor
    for j in J:
        m.Add(sum(x[i, j] for i in I) == 1)

    # Mentor capacity
    for i in I:
        m.Add(sum(x[i, j] for j in J) <= max(0, caps[i]))

    # Objective terms
    major_sum = sum(major[i][j] * x[i, j] for i in I for j in J)

    extras_scaled = [[int(round(1000 * extras[i][j])) for j in J] for i in I]
    extras_sum = sum(extras_scaled[i][j] * x[i, j] for i in I for j in J)

    return m, x, major_sum, extras_sum

@app.post("/solve")
def solve(req: SolveRequest) -> Dict[str, Any]:
    mentors = req.mentors
    mentees = req.mentees
    I = list(range(len(mentors)))
    J = list(range(len(mentees)))

    # Precompute scores
    major = [[major_match(mentors[i].major, mentees[j].target_major) for j in J] for i in I]
    extras = [[extras_score(mentors[i].extracurriculars, mentees[j].extracurriculars, req.scoring.extras_metric) for j in J] for i in I]
    caps = [m.capacity for m in mentors]

    results = []
    load = [0] * len(I)
    mm_total = 0
    ex_total = 0.0

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0

    if req.scoring.priority_mode == "lexicographic":
        # Pass 1: maximize major matches
        model1, x1, major_sum1, extras_sum1 = build_model(major, extras, caps)
        model1.Maximize(major_sum1)
        solver.Solve(model1)
        best_major = int(solver.Value(major_sum1))

        # Pass 2: fix major, then maximize extras
        model2, x2, major_sum2, extras_sum2 = build_model(major, extras, caps)
        model2.Add(major_sum2 == best_major)
        model2.Maximize(extras_sum2)
        solver.Solve(model2)

        chosen_x = x2
    else:
        # Weighted single-stage
        M = req.scoring.bigM
        model, x, major_sum, extras_sum = build_model(major, extras, caps)
        model.Maximize(M * major_sum + extras_sum)
        solver.Solve(model)
        chosen_x = x

    # Extract solution
    for i in I:
        for j in J:
            if solver.Value(chosen_x[i, j]) == 1:
                mm = (major[i][j] == 1)
                es = extras[i][j]
                results.append({
                    "mentee": mentees[j].id,
                    "mentee_name": mentees[j].name,
                    "mentor": mentors[i].id,
                    "mentor_name": mentors[i].name,
                    "major_match": mm,
                    "extras_score": es
                })
                load[i] += 1
                mm_total += 1 if mm else 0
                ex_total += es

    return {
        "objective": {"major_matches": mm_total, "extras_total": ex_total},
        "assignments": results,
        "load_by_mentor": [
            {"mentor": mentors[i].id, "mentor_name": mentors[i].name, "mentees": load[i]}
            for i in I
        ]
    }
