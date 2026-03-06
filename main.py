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

    # ── 2. Get worker IDs with this skill ─────────────────────────────────────
    skill_res = (supabase.table("worker_skills")
                 .select("worker_id")
                 .eq("category_id", category_id)
                 .execute())

    if not skill_res.data:
        return MatchResponse(workers=[], total=0)

    worker_ids = [r["worker_id"] for r in skill_res.data]

    # ── 3. Fetch worker profiles (available only) ─────────────────────────────
    wp_res = (supabase.table("worker_profiles")
              .select("user_id, is_available, is_pro, rating, total_reviews, "
                      "hourly_rate, availability_days, latitude, longitude, service_radius_km")
              .in_("user_id", worker_ids)
              .eq("is_available", True)
              .execute())

    if not wp_res.data:
        return MatchResponse(workers=[], total=0)

    available_worker_ids = [wp["user_id"] for wp in wp_res.data]

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

    for wp in wp_res.data:
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
