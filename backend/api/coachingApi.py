# api/coachingApi.py
# Finds coaching centers using Overpass API (OpenStreetMap)
# Fixed query — much broader to catch more results
# Completely FREE — no API key, no limits

from fastapi import APIRouter
import httpx
import math

coachingRouter = APIRouter()


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return round(R * 2 * math.asin(math.sqrt(a)), 1)


@coachingRouter.get("/coaching/search")
async def searchCoaching(location: str):
    """
    Finds coaching/educational centers near a location.
    Uses broader Overpass query to catch more results.
    Returns centers sorted by distance.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            # step 1 — geocode location
            geo_res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location + " India",
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1
                },
                headers={"User-Agent": "Prism-ExamApp/1.0 contact@prism.app"}
            )
            geo_data = geo_res.json()

            if not geo_data:
                return {"message": "location_not_found", "payload": [], "coords": None}

            lat = float(geo_data[0]["lat"])
            lon = float(geo_data[0]["lon"])
            city = geo_data[0].get("display_name", location).split(",")[0]
            print(f"[Coaching] Geocoded '{location}' → {lat:.4f}, {lon:.4f}")

            # step 2 — broad Overpass query
            # 50km radius, any educational amenity + name keyword search
            radius = 50000
            overpass_query = f"""
[out:json][timeout:30];
(
  node["amenity"="school"]["name"](around:{radius},{lat},{lon});
  node["amenity"="college"]["name"](around:{radius},{lat},{lon});
  node["amenity"="university"]["name"](around:{radius},{lat},{lon});
  node["amenity"="driving_school"]["name"](around:{radius},{lat},{lon});
  node["office"="educational_institution"]["name"](around:{radius},{lat},{lon});
  node["building"="school"]["name"](around:{radius},{lat},{lon});
  way["amenity"="school"]["name"](around:{radius},{lat},{lon});
  way["amenity"="college"]["name"](around:{radius},{lat},{lon});
);
out center body 50;
"""

            overpass_res = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query}
            )

            # Overpass can return HTML error pages when busy/rate-limited
            if overpass_res.status_code != 200:
                print(f"[Coaching] Overpass returned status {overpass_res.status_code}")
                return {"message": "centers", "payload": [], "coords": {"lat": lat, "lon": lon, "city": city}}

            try:
                overpass_data = overpass_res.json()
            except Exception:
                print(f"[Coaching] Overpass returned non-JSON response (likely rate-limited)")
                return {"message": "centers", "payload": [], "coords": {"lat": lat, "lon": lon, "city": city}}

            elements = overpass_data.get("elements", [])
            print(f"[Coaching] Overpass returned {len(elements)} raw elements")

            centers = []
            seen = set()
            coaching_kw = [
                "coaching", "iit", "jee", "neet", "academy", "institute",
                "classes", "tutorial", "study", "education", "learning",
                "school", "college", "centre", "center", "foundation",
                "allen", "aakash", "fiitjee", "narayana", "sri chaitanya",
                "resonance", "motion", "career", "excel", "talent"
            ]

            for el in elements:
                tags = el.get("tags", {})
                name = tags.get("name", "").strip()

                if not name or name in seen:
                    continue

                # get coordinates
                if el.get("type") == "way":
                    c = el.get("center", {})
                    el_lat = c.get("lat", lat)
                    el_lon = c.get("lon", lon)
                else:
                    el_lat = el.get("lat", lat)
                    el_lon = el.get("lon", lon)

                dist = haversine(lat, lon, el_lat, el_lon)
                if dist > 50:
                    continue

                # include if name has coaching keyword OR is school/college
                amenity = tags.get("amenity", "")
                name_lower = name.lower()
                has_kw = any(kw in name_lower for kw in coaching_kw)
                is_edu = amenity in ["school", "college", "university"]

                if not (has_kw or is_edu):
                    continue

                seen.add(name)

                # build address
                addr_parts = []
                for k in ["addr:street", "addr:suburb", "addr:city", "addr:state"]:
                    if tags.get(k):
                        addr_parts.append(tags[k])
                address = ", ".join(addr_parts) if addr_parts else city

                # categorize
                if any(k in name_lower for k in ["iit", "jee", "neet", "coaching", "institute"]):
                    category = "🎯 JEE/NEET Coaching"
                elif amenity == "college":
                    category = "🏛️ College"
                elif amenity == "university":
                    category = "🎓 University"
                else:
                    category = "📚 School/Academy"

                centers.append({
                    "name": name,
                    "address": address,
                    "distance": dist,
                    "distanceStr": f"{dist} km",
                    "category": category,
                    "lat": el_lat,
                    "lon": el_lon,
                    "phone": tags.get("phone", ""),
                    "website": tags.get("website", ""),
                })

            # sort by distance
            centers.sort(key=lambda x: x["distance"])
            print(f"[Coaching] Returning {len(centers)} centers near {location}")
            return {
                "message": "centers",
                "payload": centers[:30],
                "coords": {"lat": lat, "lon": lon, "city": city}
            }

    except Exception as e:
        print(f"[Coaching] Error: {e}")
        return {"message": "error", "payload": [], "coords": None}


@coachingRouter.get("/coaching/geocode")
async def geocodeLocation(location: str):
    """Geocodes a location string to coordinates."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location + " India", "format": "json", "limit": 1},
                headers={"User-Agent": "Prism-ExamApp/1.0"}
            )
            data = res.json()
            if data:
                return {"message": "ok", "payload": {
                    "lat": float(data[0]["lat"]),
                    "lon": float(data[0]["lon"]),
                    "city": data[0].get("display_name", location).split(",")[0]
                }}
    except Exception as e:
        print(f"[Coaching] Geocode error: {e}")
    return {"message": "error", "payload": None}