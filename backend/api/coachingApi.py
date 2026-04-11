# api/coachingApi.py
# Finds coaching centers using Overpass API (OpenStreetMap)
# Completely FREE — no API key, no limits
# More accurate than Google for education centers

from fastapi import APIRouter
import httpx
import math

coachingRouter = APIRouter()

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two coordinates."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@coachingRouter.get("/coaching/search")
async def searchCoaching(location: str):
    """
    Finds coaching/tutoring centers near a location.
    Uses Nominatim for geocoding + Overpass for POI search.
    Completely free, no API key needed.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # step 1 — geocode the location
            geo_res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": location + " India",
                    "format": "json",
                    "limit": 1
                },
                headers={"User-Agent": "Prism-ExamApp/1.0"}
            )
            geo_data = geo_res.json()

            if not geo_data:
                return {"message": "location_not_found", "payload": []}

            lat = float(geo_data[0]["lat"])
            lon = float(geo_data[0]["lon"])
            print(f"[Coaching] Geocoded '{location}' → {lat}, {lon}")

            # step 2 — Overpass query for coaching centers (30km radius)
            radius = 30000  # 30 km
            overpass_query = f"""
            [out:json][timeout:25];
            (
              node["amenity"="school"](around:{radius},{lat},{lon});
              node["amenity"="college"](around:{radius},{lat},{lon});
              node["amenity"="tutoring"](around:{radius},{lat},{lon});
              node["office"="educational_institution"](around:{radius},{lat},{lon});
              node["name"~"coaching|IIT|JEE|NEET|academy|institute",i](around:{radius},{lat},{lon});
            );
            out body;
            """

            overpass_res = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query}
            )
            overpass_data = overpass_res.json()

            centers = []
            seen_names = set()

            for element in overpass_data.get("elements", []):
                tags = element.get("tags", {})
                name = tags.get("name", "")

                if not name or name in seen_names:
                    continue

                # filter for coaching-related names
                coaching_keywords = ["coaching", "iit", "jee", "neet", "academy",
                                      "institute", "classes", "tutorial", "study"]
                name_lower = name.lower()
                is_coaching = any(kw in name_lower for kw in coaching_keywords)
                if not is_coaching:
                    continue

                seen_names.add(name)

                elem_lat = element.get("lat", lat)
                elem_lon = element.get("lon", lon)
                dist = haversine(lat, lon, elem_lat, elem_lon)

                # build address
                addr_parts = []
                for key in ["addr:street", "addr:suburb", "addr:city"]:
                    if tags.get(key):
                        addr_parts.append(tags[key])

                centers.append({
                    "name": name,
                    "address": ", ".join(addr_parts) if addr_parts else location,
                    "distance": f"{dist:.1f} km away",
                    "type": tags.get("amenity", "Coaching Center").replace("_", " ").title(),
                    "lat": elem_lat,
                    "lon": elem_lon
                })

            # sort by distance
            centers.sort(key=lambda x: float(x["distance"].split(" ")[0]))

            print(f"[Coaching] Found {len(centers)} centers near {location}")
            return {"message": "centers", "payload": centers[:20]}

    except Exception as e:
        print(f"[Coaching] Error: {e}")
        return {"message": "error", "payload": []}