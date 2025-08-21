#!/usr/bin/env python3
import argparse, time, random, math, statistics, csv, json, sys, traceback, os
from typing import List, Dict, Any, Tuple, Optional
import requests

# ---------- Defaults (expanded) ----------
DEFAULT_MAJORS = [
    # CS/Data
    "Computer Science","Data Science","Information Systems","Software Engineering","Artificial Intelligence",
    # Eng
    "Electrical Engineering","Mechanical Engineering","Civil Engineering","Chemical Engineering","Industrial Engineering",
    # Math/Science
    "Mathematics","Statistics","Physics","Biology","Chemistry","Neuroscience","Environmental Science",
    # Business/Econ
    "Economics","Finance","Accounting","Business Administration","Marketing","Operations Research",
    # Social/Arts/Humanities
    "Psychology","Sociology","Political Science","History","Philosophy","English","Communications","Design",
]
DEFAULT_EXTRAS = [
    "ACM","Robotics","Hack Club","Math Club","Volunteering","Debate","Chess","Dance","Entrepreneurship","Music",
    "Basketball","Soccer","Baseball","Tennis","Track","Esports","Theatre","Orchestra","Marching Band","Model UN",
    "Tutoring","Peer Mentoring","TA/Grader","Research Lab","Startups","Women in Tech","NSBE","SHPE","IEEE","Toastmasters",
    "Photography","Film Club","Art Club","Entrepreneurship Club","Game Dev","Data Science Club","AI Club","Cybersecurity Club",
]

# ---------- Utilities ----------
def load_list_from_file(path: str) -> List[str]:
    """Accepts either JSON array or newline-separated file."""
    if not path: return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        data = json.loads(txt)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    # fallback: newline-separated
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

# ---------- Synthetic data ----------
def sample_extras(rng: random.Random, pool: List[str], k_min=1, k_max=3) -> List[str]:
    if not pool:
        return []
    k = rng.randint(k_min, min(k_max, max(1, len(pool))))
    # avoid sampling more than pool size
    k = min(k, len(pool))
    return rng.sample(pool, k)

def generate_payload(
    n_mentors: int,
    n_mentees: int,
    *,
    seed: int,
    mode: str,
    extras_metric: str,
    majors: List[str],
    extras_pool: List[str],
    fav_majors: List[str],
    fav_weight: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    # Build weighted major list
    fav_set = {m.lower(): m for m in fav_majors}
    weights: Dict[str, int] = {m: 1 for m in majors}
    for m in majors:
        if m.lower() in fav_set:
            weights[m] = max(1, int(fav_weight))
    weighted = [m for m, w in weights.items() for _ in range(w)]

    # capacities sum == n_mentees
    base = max(1, n_mentees // max(1, n_mentors))
    caps = [base] * n_mentors
    for i in range(n_mentees - base * n_mentors):
        caps[i % n_mentors] += 1

    mentors = []
    for i in range(n_mentors):
        mentors.append({
            "id": f"m{i+1}",
            "name": f"Mentor {i+1}",
            "major": rng.choice(weighted),
            "extracurriculars": sample_extras(rng, extras_pool),
            "capacity": caps[i],
        })

    mentees = []
    for j in range(n_mentees):
        # 70% from weighted to preserve signal, rest uniform to keep diversity
        tm = rng.choice(weighted) if rng.random() < 0.70 else rng.choice(majors)
        mentees.append({
            "id": f"s{j+1}",
            "name": f"Mentee {j+1}",
            "target_major": tm,
            "extracurriculars": sample_extras(rng, extras_pool),
        })

    return {
        "mentors": mentors,
        "mentees": mentees,
        "scoring": {
            "priority_mode": mode,          # "lexicographic" | "weighted"
            "bigM": 1000,
            "extras_metric": extras_metric, # "count" | "jaccard" | "cosine"
            "major_rule": "exact",
        },
    }

# ---------- Scoring helpers ----------
def major_match(ment_major: str, targ_major: str) -> int:
    return 1 if ment_major.strip().lower() == targ_major.strip().lower() else 0

def extras_score(a: List[str], b: List[str], metric: str) -> float:
    A, B = set(map(str.lower, a)), set(map(str.lower, b))
    if metric == "count":
        return float(len(A & B))
    if metric == "jaccard":
        u = len(A | B) or 1
        return float(len(A & B)) / u
    # cosine
    vocab = sorted(A | B)
    va = [1.0 if x in A else 0.0 for x in vocab]
    vb = [1.0 if x in B else 0.0 for x in vocab]
    dot = sum(x*y for x,y in zip(va, vb))
    na = math.sqrt(sum(x*x for x in va))
    nb = math.sqrt(sum(y*y for y in vb))
    return dot / (na*nb or 1.0)

def build_score_matrix(mentors, mentees, metric: str, bigM: float = 1000.0):
    I, J = range(len(mentors)), range(len(mentees))
    major = [[major_match(mentors[i]["major"], mentees[j]["target_major"]) for j in J] for i in I]
    extras = [[extras_score(mentors[i]["extracurriculars"], mentees[j]["extracurriculars"], metric) for j in J] for i in I]
    weighted = [[bigM*major[i][j] + extras[i][j] for j in J] for i in I]
    return major, extras, weighted

# Capacity-aware greedy (mentee random order)
def greedy_baseline(mentors, mentees, *, metric: str = "count", bigM: float = 1000.0, seed: int = 0):
    rng = random.Random(seed)
    I, J = len(mentors), len(mentees)
    caps = [m["capacity"] for m in mentors]
    major, extras, weighted = build_score_matrix(mentors, mentees, metric, bigM)
    order = list(range(J))
    rng.shuffle(order)
    assign = []  # (i,j,major,extras)
    for j in order:
        best_i, best_s = None, -1e18
        for i in range(I):
            if caps[i] <= 0: continue
            s = weighted[i][j]
            if s > best_s:
                best_s, best_i = s, i
        if best_i is None:
            continue
        assign.append((best_i, j, major[best_i][j], extras[best_i][j]))
        caps[best_i] -= 1
    return assign

# ---------- Metrics ----------
def pctl(xs: List[float], q: float) -> float:
    if not xs: return 0.0
    xs = sorted(xs)
    idx = int(round(q * (len(xs)-1)))
    return xs[idx]

def mean_stderr_ci(xs: List[float], alpha=0.05) -> Tuple[float,float,float]:
    if not xs: return (0.0, 0.0, 0.0)
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) < 2 else statistics.stdev(xs)
    from math import sqrt
    se = s / sqrt(max(1, len(xs)))
    z = 1.96  # 95% normal approx
    return (m, m - z*se, m + z*se)

def same_major_pct_api(assignments: List[Dict[str, Any]], mentee_count: int) -> float:
    if mentee_count <= 0: return 0.0
    matches = sum(1 for a in assignments if a.get("major_match"))
    return 100.0 * matches / mentee_count

def extras_total_api(assignments: List[Dict[str, Any]]) -> float:
    return sum(float(a.get("extras_score", 0.0)) for a in assignments)

def extras_avg_per_match_api(assignments: List[Dict[str, Any]], mentee_count: int) -> float:
    n = len(assignments) if assignments else mentee_count
    n = max(1, n)
    return extras_total_api(assignments) / n

def same_major_pct_greedy(assign, mentee_count: int) -> float:
    if mentee_count <= 0: return 0.0
    matches = sum(1 for (_,_,mm,_) in assign if mm)
    return 100.0 * matches / mentee_count

def extras_total_greedy(assign) -> float:
    return sum(float(es) for (_,_,_,es) in assign)

def extras_avg_per_match_greedy(assign, mentee_count: int) -> float:
    n = max(1, mentee_count)
    return extras_total_greedy(assign) / n

# ---------- Cost ----------
CPU_PRICE_PER_VCPU_S   = 0.000024
MEM_PRICE_PER_GIB_S    = 0.0000025
REQ_PRICE_PER_1M       = 0.40

def estimate_cost_per_run(latency_s: float, vcpus: float, mem_gib: float) -> float:
    cpu_cost = vcpus * latency_s * CPU_PRICE_PER_VCPU_S
    mem_cost = mem_gib * latency_s * MEM_PRICE_PER_GIB_S
    req_cost = REQ_PRICE_PER_1M / 1_000_000
    return cpu_cost + mem_cost + req_cost

# ---------- Solver metric extraction ----------
def extract_solver_details(resp: Dict[str,Any]) -> Dict[str, Optional[float]]:
    m = resp.get("metrics", {}).get("solver", {})
    out = {
        "mode": m.get("mode"),
        "status": m.get("status"),
        "wall_time_s": m.get("wall_time_s"),
        "mip_gap": m.get("mip_gap"),
        "stage1_status": None, "stage1_wall_s": None, "stage1_gap": None,
        "stage2_status": None, "stage2_wall_s": None, "stage2_gap": None,
    }
    if "stage1" in m:
        s1 = m["stage1"]
        out["stage1_status"] = s1.get("status")
        out["stage1_wall_s"] = s1.get("wall_time_s")
        g1 = s1.get("mip_gap")
        if g1 is None and s1.get("objective") is not None and s1.get("best_bound") is not None:
            obj = float(s1["objective"]); bnd = float(s1["best_bound"])
            denom = max(1.0, abs(obj)); g1 = abs(obj - bnd) / denom
        out["stage1_gap"] = g1
    if "stage2" in m:
        s2 = m["stage2"]
        out["stage2_status"] = s2.get("status")
        out["stage2_wall_s"] = s2.get("wall_time_s")
        g2 = s2.get("mip_gap")
        if g2 is None and s2.get("objective") is not None and s2.get("best_bound") is not None:
            obj = float(s2["objective"]); bnd = float(s2["best_bound"])
            denom = max(1.0, abs(obj)); g2 = abs(obj - bnd) / denom
        out["stage2_gap"] = g2
    # Backfill from nested if flat missing
    if out["status"] is None:
        for k in ("single_stage","stage2","stage1"):
            if k in m and isinstance(m[k], dict):
                s = m[k]
                out["status"] = s.get("status")
                out["wall_time_s"] = s.get("wall_time_s")
                g = s.get("mip_gap")
                if g is None and s.get("objective") is not None and s.get("best_bound") is not None:
                    obj = float(s["objective"]); bnd = float(s["best_bound"])
                    denom = max(1.0, abs(obj)); g = abs(obj - bnd) / denom
                out["mip_gap"] = g
                break
    return out

# ---------- HTTP call with retry/backoff ----------
def call_api_with_retry(url: str, payload: Dict[str, Any], timeout_s: int = 300, max_retries: int = 3) -> Tuple[float, Dict[str, Any]]:
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        t0 = time.perf_counter()
        try:
            r = requests.post(url.rstrip("/") + "/solve", json=payload, timeout=timeout_s)
            t1 = time.perf_counter()
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {r.text}", response=r)
            r.raise_for_status()
            return (t1 - t0), r.json()
        except requests.RequestException as e:
            last_exc = e
            code = getattr(e.response, "status_code", None)
            if code and code in (400, 401, 403, 404, 422):
                raise
            sleep_s = min(45.0, (2 ** attempt) + random.random())
            time.sleep(sleep_s)
            attempt += 1
    if last_exc:
        raise last_exc
    raise RuntimeError("unknown error without exception")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Benchmark PairWise /solve with greedy baseline (v4, expanded data & MIP gap).")
    ap.add_argument("--url", required=True, help="https://<your-cloud-run>.run.app")
    ap.add_argument("--sizes", default="50,100,200,300,400,600,800,1000", help="Comma-separated mentee sizes; mentors ≈ mentees/2")
    ap.add_argument("--trials", type=int, default=40, help="Trials per size")
    ap.add_argument("--mode", choices=["lexicographic","weighted"], default="lexicographic")
    ap.add_argument("--extras", choices=["count","jaccard","cosine"], default="count")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--vcpu", type=float, default=1.0)
    ap.add_argument("--mem-gb", type=float, default=1.0)
    ap.add_argument("--delay-ms", type=int, default=300, help="Sleep between trials (ms)")
    # NEW: data overrides & weighting
    ap.add_argument("--majors-file", default="", help="Optional file with majors (JSON array or newline list)")
    ap.add_argument("--extras-file", default="", help="Optional file with extracurriculars (JSON array or newline list)")
    ap.add_argument("--fav-majors", default="Computer Science,Data Science", help="CSV of favored majors for weighting")
    ap.add_argument("--fav-weight", type=int, default=3, help="Weight multiplier for favored majors")
    ap.add_argument("--trials-csv", default="bench_trials.csv")
    ap.add_argument("--summary-csv", default="bench_summary.csv")
    args = ap.parse_args()

    # Build pools
    majors = load_list_from_file(args.majors_file) or DEFAULT_MAJORS
    extras_pool = load_list_from_file(args.extras_file) or DEFAULT_EXTRAS
    fav_majors = parse_csv_list(args.fav_majors)
    if not majors: 
        raise RuntimeError("Majors list is empty after loading overrides.")
    if not extras_pool:
        raise RuntimeError("Extras list is empty after loading overrides.")

    random.seed(args.seed)
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    delay_s = max(0.0, args.delay_ms / 1000.0)

    # Per-trial CSV
    trials_fields = [
        "mentors","mentees","trial_idx",
        "e2e_s","solver_wall_s","solver_status","mip_gap",
        "stage1_status","stage1_wall_s","stage1_gap",
        "stage2_status","stage2_wall_s","stage2_gap",
        "api_same_major_pct","greedy_same_major_pct","delta_same_major_pct",
        "api_extras_total","greedy_extras_total","delta_extras_total",
        "api_extras_avg","greedy_extras_avg","delta_extras_avg",
        "http_status","error"
    ]
    trial_f = open(args.trials_csv, "w", newline="", encoding="utf-8")
    trial_w = csv.DictWriter(trial_f, fieldnames=trials_fields)
    trial_w.writeheader()

    summary_rows = []
    for n_mentees in sizes:
        n_mentors = max(1, n_mentees // 2)
        print(f"\n=== Size: mentors={n_mentors}, mentees={n_mentees} | trials={args.trials} | mode={args.mode} ===")

        latencies, solver_walls, gaps = [], [], []
        api_same, greedy_same = [], []
        api_extras_tot, greedy_extras_tot = [], []
        api_extras_avg, greedy_extras_avg = [], []
        status_counts, s1_counts, s2_counts = {}, {}, {}

        for t in range(args.trials):
            seed_t = args.seed + 31*t + 7*n_mentees
            payload = generate_payload(
                n_mentors, n_mentees,
                seed=seed_t, mode=args.mode, extras_metric=args.extras,
                majors=majors, extras_pool=extras_pool,
                fav_majors=fav_majors, fav_weight=args.fav_weight,
            )
            g_assign = greedy_baseline(payload["mentors"], payload["mentees"], metric=args.extras, bigM=1000.0, seed=seed_t+99)
            g_same = same_major_pct_greedy(g_assign, n_mentees)
            g_extras_total = extras_total_greedy(g_assign)
            g_extras_avg = extras_avg_per_match_greedy(g_assign, n_mentees)

            row = {
                "mentors": n_mentors, "mentees": n_mentees, "trial_idx": t+1,
                "http_status": 200, "error": ""
            }
            try:
                e2e, resp = call_api_with_retry(args.url, payload, timeout_s=300)
                a_same = same_major_pct_api(resp.get("assignments", []), n_mentees)
                a_extras_total = extras_total_api(resp.get("assignments", []))
                a_extras_avg = extras_avg_per_match_api(resp.get("assignments", []), n_mentees)
                det = extract_solver_details(resp)

                latencies.append(e2e)
                api_same.append(a_same); greedy_same.append(g_same)
                api_extras_tot.append(a_extras_total); greedy_extras_tot.append(g_extras_total)
                api_extras_avg.append(a_extras_avg); greedy_extras_avg.append(g_extras_avg)
                if det.get("wall_time_s") is not None: solver_walls.append(det["wall_time_s"])
                if det.get("mip_gap") is not None: gaps.append(det["mip_gap"])
                if det.get("status"): status_counts[det["status"]] = status_counts.get(det["status"], 0) + 1
                if det.get("stage1_status"): s1_counts[det["stage1_status"]] = s1_counts.get(det["stage1_status"], 0) + 1
                if det.get("stage2_status"): s2_counts[det["stage2_status"]] = s2_counts.get(det["stage2_status"], 0) + 1

                print(
                    f"  trial {t+1:>2}/{args.trials}: "
                    f"e2e={e2e:6.2f}s  "
                    f"SAME_MAJOR api={a_same:5.1f}% vs greedy={g_same:5.1f}%  "
                    f"EXTRAS(avg) api={a_extras_avg:4.2f} vs greedy={g_extras_avg:4.2f}  "
                    f"status={det.get('status')} s1={det.get('stage1_status')} s2={det.get('stage2_status')} "
                    f"gap={det.get('mip_gap') if det.get('mip_gap') is not None else 'NA'}"
                )

                row.update({
                    "e2e_s": round(e2e,3),
                    "solver_wall_s": round(det["wall_time_s"],3) if det.get("wall_time_s") is not None else "",
                    "solver_status": det.get("status") or "",
                    "mip_gap": round(det["mip_gap"],6) if det.get("mip_gap") is not None else "",
                    "stage1_status": det.get("stage1_status") or "",
                    "stage1_wall_s": round(det["stage1_wall_s"],3) if det.get("stage1_wall_s") is not None else "",
                    "stage1_gap": round(det["stage1_gap"],6) if det.get("stage1_gap") is not None else "",
                    "stage2_status": det.get("stage2_status") or "",
                    "stage2_wall_s": round(det["stage2_wall_s"],3) if det.get("stage2_wall_s") is not None else "",
                    "stage2_gap": round(det["stage2_gap"],6) if det.get("stage2_gap") is not None else "",
                    "api_same_major_pct": round(a_same,1),
                    "greedy_same_major_pct": round(g_same,1),
                    "delta_same_major_pct": round(a_same - g_same,1),
                    "api_extras_total": round(a_extras_total,2),
                    "greedy_extras_total": round(g_extras_total,2),
                    "delta_extras_total": round(a_extras_total - g_extras_total,2),
                    "api_extras_avg": round(a_extras_avg,3),
                    "greedy_extras_avg": round(g_extras_avg,3),
                    "delta_extras_avg": round(a_extras_avg - g_extras_avg,3),
                })
            except requests.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                text = getattr(e.response, "text", "")[:200]
                row["http_status"] = code or 0
                row["error"] = f"HTTP {code}: {text}"
                print(f"  HTTP {code}: {text}")
            except Exception as e:
                row["http_status"] = 0
                row["error"] = f"EXC: {e}"
                print("  Exception:", e)
                traceback.print_exc()

            # persist the row
            trial_w.writerow(row)
            trial_f.flush()

            if delay_s > 0:
                time.sleep(delay_s)

        if not latencies:
            continue

        # Aggregates
        p50 = statistics.median(latencies)
        p90 = pctl(latencies, 0.90)
        p95 = pctl(latencies, 0.95)
        (sm_api_mean, sm_api_lo, sm_api_hi) = mean_stderr_ci(api_same)
        (sm_gre_mean, sm_gre_lo, sm_gre_hi) = mean_stderr_ci(greedy_same)
        delta_same = [a-b for a,b in zip(api_same, greedy_same)]
        (d_mean, d_lo, d_hi) = mean_stderr_ci(delta_same)
        (ex_api_tot_mean, _, _) = mean_stderr_ci(api_extras_tot)
        (ex_gre_tot_mean, _, _) = mean_stderr_ci(greedy_extras_tot)
        (ex_api_avg_mean, _, _) = mean_stderr_ci(api_extras_avg)
        (ex_gre_avg_mean, _, _) = mean_stderr_ci(greedy_extras_avg)
        solver_wall_p50 = statistics.median(solver_walls) if solver_walls else ""
        solver_wall_p95 = pctl(solver_walls, 0.95) if solver_walls else ""
        gap_mean = (sum(gaps)/len(gaps)) if gaps else ""

        est_cost_per_run = estimate_cost_per_run(p95, args.vcpu, args.mem_gb)

        print(
            f"  -> e2e p50={p50:.2f}s  p90={p90:.2f}s  p95={p95:.2f}s"
            f" | SAME_MAJOR mean: api={sm_api_mean:.1f}% ({sm_api_lo:.1f},{sm_api_hi:.1f}) vs greedy={sm_gre_mean:.1f}%"
            f" | Δsame mean={d_mean:.1f}pts CI[{d_lo:.1f},{d_hi:.1f}]"
            f" | EXTRAS(avg per match): api={ex_api_avg_mean:.2f} vs greedy={ex_gre_avg_mean:.2f}"
            f" | solver_wall p50={solver_wall_p50 if solver_wall_p50=='' else f'{solver_wall_p50:.2f}s'} p95={solver_wall_p95 if solver_wall_p95=='' else f'{solver_wall_p95:.2f}s'}"
            f" | avg gap={gap_mean if gap_mean=='' else f'{gap_mean:.4f}'}"
            f" | cost@p95=${est_cost_per_run:.6f}"
            f" | stage1={s1_counts} stage2={s2_counts} overall={status_counts}"
        )

        summary_rows.append({
            "mentors": n_mentors, "mentees": n_mentees, "trials": len(latencies),
            "e2e_p50_s": round(p50,3), "e2e_p90_s": round(p90,3), "e2e_p95_s": round(p95,3),
            "solver_wall_p50_s": round(solver_wall_p50,3) if solver_wall_p50!="" else "",
            "solver_wall_p95_s": round(solver_wall_p95,3) if solver_wall_p95!="" else "",
            "same_major_api_mean_pct": round(sm_api_mean,2),
            "same_major_api_ci_lo": round(sm_api_lo,2), "same_major_api_ci_hi": round(sm_api_hi,2),
            "same_major_greedy_mean_pct": round(sm_gre_mean,2),
            "delta_same_major_mean_pts": round(d_mean,2),
            "delta_same_major_ci_lo": round(d_lo,2), "delta_same_major_ci_hi": round(d_hi,2),
            "extras_total_api_mean": round(ex_api_tot_mean,2),
            "extras_total_greedy_mean": round(ex_gre_tot_mean,2),
            "extras_avg_api_mean": round(ex_api_avg_mean,3),
            "extras_avg_greedy_mean": round(ex_gre_avg_mean,3),
            "avg_mip_gap": round(gap_mean,6) if gap_mean!="" else "",
            "est_cost_per_run_usd_at_p95": round(est_cost_per_run,6),
            "stage1_status_counts_json": json.dumps(s1_counts),
            "stage2_status_counts_json": json.dumps(s2_counts),
            "overall_status_counts_json": json.dumps(status_counts),
        })

    # Close CSVs
    trial_f.close()
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "mentors","mentees","trials",
            "e2e_p50_s","e2e_p90_s","e2e_p95_s",
            "solver_wall_p50_s","solver_wall_p95_s",
            "same_major_api_mean_pct","same_major_api_ci_lo","same_major_api_ci_hi",
            "same_major_greedy_mean_pct",
            "delta_same_major_mean_pts","delta_same_major_ci_lo","delta_same_major_ci_hi",
            "extras_total_api_mean","extras_total_greedy_mean",
            "extras_avg_api_mean","extras_avg_greedy_mean",
            "avg_mip_gap","est_cost_per_run_usd_at_p95",
            "stage1_status_counts_json","stage2_status_counts_json","overall_status_counts_json"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"\nSaved per-trial → {args.trials_csv}")
    print(f"Saved summary  → {args.summary_csv}")

if __name__ == "__main__":
    main()
