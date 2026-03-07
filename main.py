# gigmatcher-api/main.py
# GigMatcher FastAPI Matching Engine
#
# Endpoints:
#   POST /match   — find + rank available workers for a job
#   GET  /health  — health check for Render
#
# Deploy on Render:
#   1. Create new Web Service → connect GitHub repo
#   2. Root directory: gigmatcher-api
#   3. Build command: pip install -r requirements.txt
#   4. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
#   5. Set env vars: SUPABASE_URL, SUPABASE_SERVICE_KEY

import os
import math
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GigMatcher Matching Engine",
    description="Worker matching API for GigMatcher",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase client (service role — bypasses RLS for server-side queries) ─────

def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")   # service_role key, NOT anon key
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)

# ── Haversine distance ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Returns distance in km between two lat/lng coordinates."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(d_lng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def format_distance(km: float) -> str:
    if km < 1.0:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"

# ── Match score ───────────────────────────────────────────────────────────────
# Weighted score used when sort=rating (default):
#   50% rating (0–5 scale normalised to 0–1)
#   30% proximity (closer = higher score, capped at 20km)
#   20% Pro badge bonus

def match_score(rating: float, distance_km: Optional[float], is_pro: bool) -> float:
    rating_score    = (rating / 5.0) * 0.50
    proximity_score = 0.0
    if distance_km is not None:
        capped = min(distance_km, 20.0)
        proximity_score = ((20.0 - capped) / 20.0) * 0.30
    pro_score = 0.20 if is_pro else 0.0
    return rating_score + proximity_score + pro_score

# ── Request / Response models ─────────────────────────────────────────────────

class MatchRequest(BaseModel):
    category_slug:  str
    required_tools: list[str] = []
    sort:           str = "rating"          # "rating" | "distance" | "price"
    customer_lat:   Optional[float] = None  # customer's job location
    customer_lng:   Optional[float] = None

class WorkerResult(BaseModel):
    id:               str
    name:             str
    photo:            str
    rating:           float
    review_count:     int
    distance:         str                   # "1.2 km" or "—"
    distance_km:      Optional[float]       # raw km for frontend sorting
    rate:             int
    skills:           list[str]
    tools:            list[str]
    has_all_tools:    bool
    missing_tool_count: int
    available:        bool
    availability_days: list[bool]
    is_pro:           bool

class MatchResponse(BaseModel):
    workers: list[WorkerResult]
    total:   int

# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "gigmatcher-api"}

# ── /predict-demand ───────────────────────────────────────────────────────────
# Recalculates demand predictions from real job history and writes results
# to the demand_predictions table. Called:
#   - On a schedule (Render cron or external ping every hour)
#   - Manually via /docs for testing
#
# Algorithm:
#   1. Fetch all jobs from last 30 days with lat/lng + category
#   2. Bucket by 0.05° grid (~5km squares)
#   3. Score = (7d count × 3) + (30d count × 1), normalised 0-100
#   4. Clear stale predictions, insert fresh ones
#   5. Returns top 5 predictions for verification

class DemandPrediction(BaseModel):
    category_name:          str
    area_name:              str
    area_lat:               float
    area_lng:               float
    predicted_demand_score: float
    job_count_7d:           int
    job_count_30d:          int

class DemandResponse(BaseModel):
    predictions: list[DemandPrediction]
    total:       int
    computed_at: str

@app.post("/predict-demand", response_model=DemandResponse)
async def predict_demand():
    logger.info("Starting demand prediction computation")
    supabase = get_supabase()

    from datetime import datetime, timezone, timedelta

    now    = datetime.now(timezone.utc)
    ago_7d  = (now - timedelta(days=7)).isoformat()
    ago_30d = (now - timedelta(days=30)).isoformat()

    # ── 1. Fetch jobs with location from last 30 days ─────────────────────────
    jobs_res = (supabase.table("jobs")
                .select("category_id, latitude, longitude, created_at, status")
                .gte("created_at", ago_30d)
                .not_("latitude",  "is", "null")
                .not_("longitude", "is", "null")
                .execute())

    jobs = jobs_res.data or []
    logger.info(f"Fetched {len(jobs)} jobs with location from last 30 days")

    if not jobs:
        # No real data — call DB function to keep seeded data fresh
        supabase.rpc("compute_demand_predictions").execute()
        return DemandResponse(predictions=[], total=0, computed_at=now.isoformat())

    # ── 2. Fetch category names ───────────────────────────────────────────────
    cat_res = supabase.table("service_categories").select("id, name").execute()
    cat_map = {c["id"]: c["name"] for c in (cat_res.data or [])}

    # ── 3. Bucket jobs by category + 0.05° grid ───────────────────────────────
    BUCKET_SIZE = 0.05  # ~5km
    from collections import defaultdict

    buckets: dict[tuple, dict] = defaultdict(lambda: {"count_7d": 0, "count_30d": 0})

    for j in jobs:
        cat_id = j.get("category_id")
        lat    = j.get("latitude")
        lng    = j.get("longitude")
        if not cat_id or lat is None or lng is None:
            continue

        bucket_lat = round(round(float(lat) / BUCKET_SIZE) * BUCKET_SIZE, 4)
        bucket_lng = round(round(float(lng) / BUCKET_SIZE) * BUCKET_SIZE, 4)
        key = (cat_id, bucket_lat, bucket_lng)

        buckets[key]["count_30d"] += 1
        if j.get("created_at", "") >= ago_7d:
            buckets[key]["count_7d"] += 1

    logger.info(f"Found {len(buckets)} demand buckets")

    # ── 4. Score and build prediction rows ────────────────────────────────────
    predictions_to_insert = []
    prediction_results    = []

    for (cat_id, blat, blng), counts in buckets.items():
        score = min(100.0, (counts["count_7d"] * 3 + counts["count_30d"] * 1) * 10)
        cat_name = cat_map.get(cat_id, "Service")
        area_name = f"{cat_name} — {blat:.2f}°N {blng:.2f}°E"

        predictions_to_insert.append({
            "category_id":            cat_id,
            "area_name":              area_name,
            "area_lat":               blat,
            "area_lng":               blng,
            "predicted_demand_score": score,
            "job_count_7d":           counts["count_7d"],
            "job_count_30d":          counts["count_30d"],
            "computed_at":            now.isoformat(),
        })
        prediction_results.append(DemandPrediction(
            category_name=cat_name, area_name=area_name,
            area_lat=blat, area_lng=blng,
            predicted_demand_score=score,
            job_count_7d=counts["count_7d"],
            job_count_30d=counts["count_30d"],
        ))

    # ── 5. Clear stale, insert fresh ─────────────────────────────────────────
    # Only clear rows computed by FastAPI (keep seeded rows if no real data)
    stale_cutoff = (now - timedelta(hours=2)).isoformat()
    supabase.table("demand_predictions").delete().lt("computed_at", stale_cutoff).execute()

    if predictions_to_insert:
        supabase.table("demand_predictions").insert(predictions_to_insert).execute()
        logger.info(f"Inserted {len(predictions_to_insert)} demand predictions")

    # Sort by score descending for response
    prediction_results.sort(key=lambda p: p.predicted_demand_score, reverse=True)

    return DemandResponse(
        predictions=prediction_results[:10],
        total=len(prediction_results),
        computed_at=now.isoformat(),
    )

# ── /demand-summary ───────────────────────────────────────────────────────────
# Lightweight read used by worker dashboard to get top demand alert.
# Returns the single highest-scoring prediction for a given worker location.

@app.get("/demand-summary")
async def demand_summary(
    worker_lat: Optional[float] = None,
    worker_lng: Optional[float] = None,
):
    supabase = get_supabase()

    res = (supabase.table("demand_predictions")
           .select("area_name, predicted_demand_score, area_lat, area_lng, "
                   "job_count_7d, category_id")
           .order("predicted_demand_score", desc=True)
           .limit(10)
           .execute())

    predictions = res.data or []
    if not predictions:
        return {"alert": "", "top_predictions": []}

    # If worker location given, find closest high-demand area
    best = predictions[0]
    if worker_lat is not None and worker_lng is not None:
        for p in predictions:
            if p.get("area_lat") and p.get("area_lng"):
                dist = haversine_km(worker_lat, worker_lng,
                                    float(p["area_lat"]), float(p["area_lng"]))
                if dist <= 10:   # within 10km
                    best = p
                    break

    # Fetch category name
    cat_name = ""
    if best.get("category_id"):
        cat_res = (supabase.table("service_categories")
                   .select("name")
                   .eq("id", best["category_id"])
                   .single()
                   .execute())
        cat_name = cat_res.data.get("name", "") if cat_res.data else ""

    score    = best.get("predicted_demand_score", 0)
    area     = best.get("area_name", "your area")
    jobs_7d  = best.get("job_count_7d", 0)

    if score >= 80:
        level = "🔥 Very high"
    elif score >= 50:
        level = "📈 High"
    else:
        level = "📊 Moderate"

    alert = (f"{level} demand for {cat_name} in {area}!"
             if cat_name else
             f"{level} demand in {area}!")
    if jobs_7d > 0:
        alert += f" ({jobs_7d} jobs this week)"

    return {"alert": alert, "top_predictions": predictions[:5]}

# ── /match ────────────────────────────────────────────────────────────────────

@app.post("/match", response_model=MatchResponse)
async def match_workers(req: MatchRequest):
    logger.info(f"Match request: category={req.category_slug} sort={req.sort} "
                f"lat={req.customer_lat} lng={req.customer_lng}")

    supabase = get_supabase()

    # ── 1. Resolve category name from slug ────────────────────────────────────
    SLUG_TO_LABEL = {
        "plumber":     "Plumber",
        "electrician": "Electrician",
        "carpenter":   "Carpenter",
        "tailor":      "Tailor",
        "mechanic":    "Mechanic",
        "painter":     "Painter",
    }
    category_label = SLUG_TO_LABEL.get(req.category_slug)
    if not category_label:
        raise HTTPException(status_code=400, detail=f"Unknown category: {req.category_slug}")

    cat_res = (supabase.table("service_categories")
               .select("id")
               .eq("name", category_label)
               .single()
               .execute())

    if not cat_res.data:
        raise HTTPException(status_code=404, detail=f"Category '{category_label}' not found in DB")

    category_id = cat_res.data["id"]
    logger.info(f"Step 1: category_id={category_id} for label={category_label}")

    # ── 2. Get worker IDs with this skill ─────────────────────────────────────
    skill_res = (supabase.table("worker_skills")
                 .select("worker_id")
                 .eq("category_id", category_id)
                 .execute())

    logger.info(f"Step 2: worker_skills returned {len(skill_res.data or [])} rows")

    if not skill_res.data:
        return MatchResponse(workers=[], total=0)

    worker_ids = [r["worker_id"] for r in skill_res.data]

    # ── 3. Fetch worker profiles — filter available in Python (supabase-py bool bug) ─
    wp_res = (supabase.table("worker_profiles")
              .select("user_id, is_available, is_pro, rating, total_reviews, "
                      "hourly_rate, availability_days, latitude, longitude, service_radius_km")
              .in_("user_id", worker_ids)
              .execute())

    logger.info(f"Step 3: worker_profiles returned {len(wp_res.data or [])} rows (before availability filter)")

    # Filter available workers in Python — avoids supabase-py boolean comparison issues
    available_profiles = [wp for wp in (wp_res.data or []) if wp.get("is_available") is True]
    logger.info(f"Step 3: {len(available_profiles)} workers are available")

    if not available_profiles:
        return MatchResponse(workers=[], total=0)

    available_worker_ids = [wp["user_id"] for wp in available_profiles]

    # ── 4. Fetch profiles, tools, skills in parallel ──────────────────────────
    profiles_res = (supabase.table("profiles")
                    .select("id, full_name, profile_photo_url")
                    .in_("id", available_worker_ids)
                    .execute())

    tools_res = (supabase.table("worker_tools")
                 .select("worker_id, tool_name")
                 .in_("worker_id", available_worker_ids)
                 .execute())

    skills_res = (supabase.table("worker_skills")
                  .select("worker_id, service_categories(name)")
                  .in_("worker_id", available_worker_ids)
                  .execute())

    # ── 5. Index data for O(1) lookup ─────────────────────────────────────────
    profiles_map = {p["id"]: p for p in (profiles_res.data or [])}

    tools_map: dict[str, list[str]] = {}
    for t in (tools_res.data or []):
        tools_map.setdefault(t["worker_id"], []).append(t["tool_name"])

    skills_map: dict[str, list[str]] = {}
    for s in (skills_res.data or []):
        cat = s.get("service_categories")
        if isinstance(cat, dict):
            name = cat.get("name", "")
        elif isinstance(cat, list) and cat:
            name = cat[0].get("name", "")
        else:
            name = ""
        if name:
            skills_map.setdefault(s["worker_id"], []).append(name)

    required_tools_lower = [t.lower() for t in req.required_tools]

    # ── 6. Build worker results ───────────────────────────────────────────────
    results: list[WorkerResult] = []

    for wp in available_profiles:
        wid = wp["user_id"]

        # Distance calculation
        distance_km: Optional[float] = None
        w_lat = wp.get("latitude")
        w_lng = wp.get("longitude")

        if (req.customer_lat is not None and req.customer_lng is not None
                and w_lat is not None and w_lng is not None):
            distance_km = haversine_km(
                float(req.customer_lat), float(req.customer_lng),
                float(w_lat), float(w_lng)
            )

            # Filter by worker's service radius (if set)
            radius = wp.get("service_radius_km")
            if radius is not None and distance_km > float(radius):
                logger.debug(f"Worker {wid} excluded: distance {distance_km:.1f}km > radius {radius}km")
                continue

        profile       = profiles_map.get(wid, {})
        worker_tools  = tools_map.get(wid, [])
        worker_skills = skills_map.get(wid, [])

        worker_tools_lower = [t.lower() for t in worker_tools]
        missing_tools = [t for t in required_tools_lower if t not in worker_tools_lower]

        availability_days = wp.get("availability_days")
        if not isinstance(availability_days, list):
            availability_days = [True] * 7

        results.append(WorkerResult(
            id=wid,
            name=profile.get("full_name", "Worker"),
            photo=profile.get("profile_photo_url", "") or "",
            rating=float(wp.get("rating") or 0),
            review_count=int(wp.get("total_reviews") or 0),
            distance=format_distance(distance_km) if distance_km is not None else "—",
            distance_km=round(distance_km, 2) if distance_km is not None else None,
            rate=int(wp.get("hourly_rate") or 0),
            skills=worker_skills,
            tools=worker_tools,
            has_all_tools=len(missing_tools) == 0,
            missing_tool_count=len(missing_tools),
            available=bool(wp.get("is_available", False)),
            availability_days=availability_days,
            is_pro=bool(wp.get("is_pro", False)),
        ))

    # ── 7. Sort ───────────────────────────────────────────────────────────────
    if req.sort == "distance":
        # Workers without distance go to bottom
        results.sort(key=lambda w: (w.distance_km is None, w.distance_km or 0))
    elif req.sort == "price":
        results.sort(key=lambda w: w.rate)
    else:
        # Default: match score (rating + proximity + pro)
        results.sort(
            key=lambda w: match_score(w.rating, w.distance_km, w.is_pro),
            reverse=True
        )

    logger.info(f"Returning {len(results)} workers for category={req.category_slug}")
    return MatchResponse(workers=results, total=len(results))
