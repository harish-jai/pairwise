from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from collections import Counter
from ortools.sat.python import cp_model
import math, os, json, time, logging

# -------------------- FastAPI & CORS --------------------
app = FastAPI()

def _parse_origins(val: str) -> List[str]:
    if not val:
        return []
    val = val.strip()
    if val.startswith("["):
        try:
            arr = json.loads(val)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in val.split(",") if x.strip()]

_raw = os.getenv("_ALLOWED_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or ""
origins = _parse_origins(_raw)
origin_regex = os.getenv("_ALLOWED_ORIGIN_REGEX") or os.getenv("ALLOWED_ORIGIN_REGEX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # exact domains
    allow_origin_regex=origin_regex,  # e.g. r"https://.*\.vercel\.app$"
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Global error handler --------------------
@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logging.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})

# -------------------- Health / CORS debug --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/cors-debug")
def cors_debug():
    return {"allow_origins": origins, "allow_origin_regex": origin_regex}

# -------------------- Models --------------------
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
    major_rule: Literal["exact"] = "exact"

class SolveRequest(BaseModel):
    mentors: List[Mentor]
    mentees: List[Mentee]
    scoring: Scoring

# -------------------- Helpers --------------------
def extras_score(a: List[str], b: List[str], metric: str) -> float:
    A, B = set(map(str.lower, a)), set(map(str.lower, b))
    if metric == "count":
        return float(len(A & B))
    if metric == "jaccard":
        return float(len(A & B)) / (len(A | B) or 1)
    # cosine over multi-hot of union
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
    """
    Return (model, x, major_sum, extras_sum).
    x[i,j] ∈ {0,1}, each mentee exactly one mentor, mentor capacity respected.
    """
    I = range(len(caps))
    Jn = len(extras[0]) if (extras and len(extras[0]) > 0) else 0
    J = range(Jn)
    m = cp_model.CpModel()
    x = {(i, j): m.NewBoolVar(f"x_{i}_{j}") for i in I for j in J}

    # Each mentee exactly one mentor
    for j in J:
        m.Add(sum(x[i, j] for i in I) == 1)

    # Mentor capacity
    for i in I:
        m.Add(sum(x[i, j] for j in J) <= max(0, caps[i]))

    # Objectives
    major_sum = sum(major[i][j] * x[i, j] for i in I for j in J)
    extras_scaled = [[int(round(1000 * extras[i][j])) for j in J] for i in I]
    extras_sum = sum(extras_scaled[i][j] * x[i, j] for i in I for j in J)
    return m, x, major_sum, extras_sum

def _rel_gap(obj: Optional[float], bound: Optional[float]) -> Optional[float]:
    if obj is None or bound is None:
        return None
    denom = max(1.0, abs(obj))
    return abs(bound - obj) / denom

def _new_solver():
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(os.getenv("MAX_SOLVE_SECS", "15"))
    s.parameters.num_search_workers = int(os.getenv("NUM_WORKERS", "1"))  # reproducible & modest CPU
    return s

# -------------------- /solve --------------------
@app.post("/solve")
def solve(req: SolveRequest) -> Dict[str, Any]:
    mentors = req.mentors
    mentees = req.mentees

    if not mentors or not mentees:
        return JSONResponse(status_code=400, content={"error": "invalid_input", "detail": "Must provide non-empty mentors and mentees."})

    I = list(range(len(mentors)))
    J = list(range(len(mentees)))

    # Precompute scores
    major = [[major_match(mentors[i].major, mentees[j].target_major) for j in J] for i in I]
    extras = [[extras_score(mentors[i].extracurriculars, mentees[j].extracurriculars, req.scoring.extras_metric) for j in J] for i in I]
    caps = [max(0, int(m.capacity)) for m in mentors]

    # Capacity sanity: total capacity must cover mentees
    total_cap = sum(caps)
    if total_cap < len(J):
        cap_by_major = Counter([m.major.strip().lower() for m in mentors for _ in range(max(0, m.capacity))])
        dem_by_major = Counter([s.target_major.strip().lower() for s in mentees])
        return JSONResponse(
            status_code=400,
            content={
                "error": "insufficient_capacity",
                "detail": "Total mentor capacity is less than number of mentees.",
                "total_capacity": total_cap,
                "mentees": len(J),
                "cap_by_major": dict(cap_by_major),
                "dem_by_major": dict(dem_by_major),
            },
        )

    results: List[Dict[str, Any]] = []
    load = [0] * len(I)
    mm_total = 0
    ex_total = 0.0
    solver_metrics: Dict[str, Any] = {}

    if req.scoring.priority_mode == "lexicographic":
        # ---------- Stage 1: maximize major matches ----------
        model1, x1, major_sum1, _ = build_model(major, extras, caps)
        solver1 = _new_solver()
        model1.Maximize(major_sum1)

        t0 = time.perf_counter()
        status1 = solver1.Solve(model1)
        wall1 = time.perf_counter() - t0

        try:
            obj1 = float(solver1.ObjectiveValue())
        except Exception:
            obj1 = None
        try:
            bound1 = float(solver1.BestObjectiveBound())
        except Exception:
            bound1 = None
        gap1 = _rel_gap(obj1, bound1)

        best_major = int(solver1.Value(major_sum1)) if status1 in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0

        solver_metrics["stage1"] = {
            "status": solver1.StatusName(status1),
            "objective": obj1,
            "best_bound": bound1,
            "mip_gap": gap1,
            "wall_time_s": wall1,
            "major_optimum": best_major,
        }

        # ---------- Stage 2: fix major optimum, maximize extras ----------
        model2, x2, major_sum2, extras_sum2 = build_model(major, extras, caps)
        model2.Add(major_sum2 == best_major)
        model2.Maximize(extras_sum2)

        solver2 = _new_solver()
        t0 = time.perf_counter()
        status2 = solver2.Solve(model2)
        wall2 = time.perf_counter() - t0

        try:
            obj2 = float(solver2.ObjectiveValue())
        except Exception:
            obj2 = None
        try:
            bound2 = float(solver2.BestObjectiveBound())
        except Exception:
            bound2 = None
        gap2 = _rel_gap(obj2, bound2)

        solver_metrics["stage2"] = {
            "status": solver2.StatusName(status2),
            "objective": obj2,
            "best_bound": bound2,
            "mip_gap": gap2,
            "wall_time_s": wall2,
        }
        # flat/summary fields so clients don’t have to dig
        solver_metrics["mode"] = "lexicographic"
        solver_metrics["status"] = solver2.StatusName(status2)
        solver_metrics["mip_gap"] = gap2                       # <— record gap
        solver_metrics["wall_time_s"] = wall1 + wall2          # total wall

        chosen_x = x2
        chosen_solver = solver2
        final_status = status2

    else:
        # ---------- Weighted single-stage ----------
        M = req.scoring.bigM
        model, x, major_sum, extras_sum = build_model(major, extras, caps)
        model.Maximize(M * major_sum + extras_sum)

        solver = _new_solver()
        t0 = time.perf_counter()
        status = solver.Solve(model)
        wall = time.perf_counter() - t0

        try:
            obj = float(solver.ObjectiveValue())
        except Exception:
            obj = None
        try:
            bound = float(solver.BestObjectiveBound())
        except Exception:
            bound = None
        gap = _rel_gap(obj, bound)

        solver_metrics["single_stage"] = {
            "status": solver.StatusName(status),
            "objective": obj,
            "best_bound": bound,
            "mip_gap": gap,
            "wall_time_s": wall,
        }
        solver_metrics["mode"] = "weighted"                    
        solver_metrics["status"] = solver.StatusName(status)   
        solver_metrics["mip_gap"] = gap                        
        solver_metrics["wall_time_s"] = wall                   

        chosen_x = x
        chosen_solver = solver
        final_status = status

    # If no feasible/optimal solution, return 422 with diagnostics (not a 500)
    if final_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return JSONResponse(
            status_code=422,
            content={
                "error": "no_feasible_solution",
                "detail": f"Solver status: {chosen_solver.StatusName(final_status)}",
                "metrics": {"solver": solver_metrics},
            },
        )

    # ---------- Extract solution (use the solver that solved the chosen model) ----------
    I_len, J_len = len(mentors), len(mentees)
    for i in range(I_len):
        for j in range(J_len):
            try:
                val = chosen_solver.Value(chosen_x[i, j])
            except KeyError:
                continue
            if val == 1:
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

    # ---------- Response ----------
    resp: Dict[str, Any] = {
        "objective": {"major_matches": mm_total, "extras_total": ex_total},
        "assignments": results,
        "load_by_mentor": [
            {"mentor": mentors[i].id, "mentor_name": mentors[i].name, "mentees": load[i]}
            for i in range(I_len)
        ],
        "metrics": {
            "size": {"mentors": I_len, "mentees": J_len},
            "quality": {
                "same_major_rate": (mm_total / (J_len or 1)),
                "avg_extras": (ex_total / (len(results) or 1)) if results else 0.0
            },
            "solver": solver_metrics
        }
    }
    return resp
