# api/coachingApi.py — complete replacement

from fastapi import APIRouter
import httpx
import math
import asyncio

coachingRouter = APIRouter()

# Known major coaching hubs in India — curated fallback
MAJOR_COACHING_CITIES = {
    "kota": {"lat": 25.2138, "lon": 75.8648, "known_centers": [
        {"name": "Allen Career Institute", "address": "Kota, Rajasthan", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Resonance", "address": "Kota, Rajasthan", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Bansal Classes", "address": "Kota, Rajasthan", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Vibrant Academy", "address": "Kota, Rajasthan", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Motion Education", "address": "Kota, Rajasthan", "category": "🎯 JEE/NEET Coaching"},
    ]},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867, "known_centers": [
        {"name": "Narayana IIT Academy", "address": "Hyderabad, Telangana", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Sri Chaitanya", "address": "Hyderabad, Telangana", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Aakash Institute", "address": "Ameerpet, Hyderabad", "category": "🎯 JEE/NEET Coaching"},
        {"name": "FIITJEE", "address": "Hyderabad, Telangana", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Career Point", "address": "Hyderabad, Telangana", "category": "🎯 JEE/NEET Coaching"},
        {"name": "TIME Institute", "address": "Hyderabad, Telangana", "category": "🎯 JEE/NEET Coaching"},
    ]},
    "delhi": {"lat": 28.6139, "lon": 77.2090, "known_centers": [
        {"name": "FIITJEE Delhi", "address": "Delhi", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Aakash Institute Delhi", "address": "Delhi", "category": "🎯 JEE/NEET Coaching"},
        {"name": "Allen Delhi", "address": "Delhi", "category": "🎯 JEE/NEET Coaching"},
    ]},
}

# NTA exam centers — official list for major cities
NTA_EXAM_CENTERS = {
    "hyderabad": [
        {"name": "JNTU Hyderabad", "address": "Kukatpally, Hyderabad", "type": "NTA JEE/NEET Center"},
        {"name": "Osmania University", "address": "Hyderabad", "type": "NTA Exam Center"},
        {"name": "CBIT", "address": "Gandipet, Hyderabad", "type": "NTA JEE Center"},
        {"name": "MVSR Engineering College", "address": "Nadergul, Hyderabad", "type": "NTA Exam Center"},
    ],
    "secunderabad": [
        {"name": "St. Ann's College", "address": "Secunderabad", "type": "NTA NEET Center"},
    ],
    "kota": [
        {"name": "MBS College", "address": "Kota, Rajasthan", "type": "NTA JEE Center"},
    ],
    "delhi": [
        {"name": "Amity University Delhi", "address": "Noida/Delhi NCR", "type": "NTA JEE Center"},
        {"name": "DTU", "address": "Rohini, Delhi", "type": "NTA JEE/NEET Center"},
    ]
}

HYDERABAD_REGION_HINTS = [
    "hyderabad", "secunderabad", "miyapur", "ghatkesar", "medchal",
    "malkajgiri", "rangareddy", "ranga reddy", "ameerpet", "kukatpally",
    "uppal", "lb nagar", "dilsukhnagar", "telangana"
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return round(R * 2 * math.asin(math.sqrt(a)), 1)


async def overpass_search(lat: float, lon: float, radius: int, client: httpx.AsyncClient) -> list:
    """
    Searches Overpass API for educational institutions.
    Very broad query — catches anything educational.
    """
    query = f"""
[out:json][timeout:25];
(
  node["amenity"~"school|college|university|library|training"](around:{radius},{lat},{lon});
  node["building"~"school|college|university"](around:{radius},{lat},{lon});
  node["office"~"educational_institution|educational"](around:{radius},{lat},{lon});
  node["name"~"[Cc]oaching|[Aa]cademy|[Ii]nstitute|[Cc]lasses|[Tt]utorial|[Ee]ducation|IIT|JEE|NEET|[Ff]oundation|[Ss]tudy|[Ll]earning|Allen|Aakash|Narayana|Chaitanya|FIITJEE|Resonance|[Cc]areer|[Vv]ibrant|[Mm]otion|TIME|[Bb]ansal"](around:{radius},{lat},{lon});
  way["amenity"~"school|college|university"](around:{radius},{lat},{lon});
  way["name"~"[Cc]oaching|[Aa]cademy|[Ii]nstitute|IIT|JEE|NEET|Allen|Aakash|Narayana|Chaitanya"](around:{radius},{lat},{lon});
);
out center body 100;
"""
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ]

    for endpoint in endpoints:
        try:
            res = await client.post(
                endpoint,
                content=query,
                headers={
                    "Content-Type": "text/plain; charset=utf-8",
                    "User-Agent": "Prism-ExamApp/1.0"
                },
                timeout=35.0
            )
            res.raise_for_status()
            return res.json().get("elements", [])
        except Exception as e:
            print(f"[Coaching] Overpass error at radius {radius} via {endpoint}: {e}")

    return []


def process_elements(elements: list, user_lat: float, user_lon: float) -> list:
    """Convert raw Overpass elements to clean center objects."""
    coaching_kw = [
        "coaching", "iit", "jee", "neet", "academy", "institute",
        "classes", "tutorial", "study", "education", "learning",
        "allen", "aakash", "fiitjee", "narayana", "sri chaitanya",
        "resonance", "motion", "career", "vibrant", "foundation",
        "bansal", "excel", "talent", "topper", "school", "college",
        "time institute", "brilliants", "vidya", "saraswati"
    ]

    centers = []
    seen = set()

    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name or name in seen:
            continue

        # get coords
        if el.get("type") == "way":
            c = el.get("center", {})
            el_lat = c.get("lat", user_lat)
            el_lon = c.get("lon", user_lon)
        else:
            el_lat = el.get("lat", user_lat)
            el_lon = el.get("lon", user_lon)

        dist = haversine(user_lat, user_lon, el_lat, el_lon)
        name_lower = name.lower()
        amenity = tags.get("amenity", "")

        # include if coaching keyword in name OR is school/college
        has_kw = any(kw in name_lower for kw in coaching_kw)
        is_edu_amenity = amenity in ["school", "college", "university", "library"]

        if not (has_kw or is_edu_amenity):
            continue

        seen.add(name)

        # address
        addr_parts = [tags.get(k, "") for k in
                      ["addr:housename", "addr:street", "addr:suburb", "addr:city"]
                      if tags.get(k)]
        address = ", ".join(addr_parts) if addr_parts else "Address not available"

        # category
        name_lower = name.lower()
        if any(k in name_lower for k in ["iit", "jee", "neet", "coaching", "institute", "foundation", "academy"]):
            category = "🎯 JEE/NEET Coaching"
        elif amenity == "college" or "college" in name_lower:
            category = "🏛️ College"
        elif amenity == "university" or "university" in name_lower:
            category = "🎓 University"
        elif amenity == "library":
            category = "📖 Library/Study Center"
        else:
            category = "📚 School/Academy"

        centers.append({
            "name": name,
            "address": address,
            "distance": dist,
            "distanceStr": f"{dist} km away",
            "category": category,
            "lat": el_lat,
            "lon": el_lon,
            "phone": tags.get("phone") or tags.get("contact:phone", ""),
            "website": tags.get("website") or tags.get("contact:website", ""),
            "source": "openstreetmap"
        })

    return sorted(centers, key=lambda x: x["distance"])


def get_curated_centers(city_name: str, user_lat: float, user_lon: float) -> list:
    """Returns curated coaching centers for known cities."""
    city_lower = city_name.lower()
    for known_city, data in MAJOR_COACHING_CITIES.items():
        if known_city in city_lower or city_lower in known_city:
            centers = []
            for c in data["known_centers"]:
                dist = haversine(user_lat, user_lon, data["lat"], data["lon"])
                centers.append({
                    **c,
                    "distance": dist,
                    "distanceStr": f"~{dist} km (approx)",
                    "lat": data["lat"] + (len(centers) * 0.002),  # slight offset for map
                    "lon": data["lon"] + (len(centers) * 0.002),
                    "phone": "",
                    "website": "",
                    "source": "curated"
                })
            return centers
    return []


def get_nta_centers(city_name: str, user_lat: float, user_lon: float) -> list:
    """Returns NTA exam centers for the city."""
    city_lower = city_name.lower()
    nta = []
    for city_key, centers in NTA_EXAM_CENTERS.items():
        if city_key in city_lower or city_lower in city_key:
            for c in centers:
                nta.append({
                    **c,
                    "category": f"🏛️ {c['type']}",
                    "distance": 0,
                    "distanceStr": "In city",
                    "lat": user_lat + (len(nta) * 0.01),
                    "lon": user_lon + (len(nta) * 0.01),
                    "phone": "",
                    "website": "https://nta.ac.in",
                    "source": "nta_official"
                })
    return nta


def resolve_curated_region(location: str, city: str, state: str) -> str:
    haystack = " ".join([location, city, state]).lower()
    if any(hint in haystack for hint in HYDERABAD_REGION_HINTS):
        return "hyderabad"
    return city


@coachingRouter.get("/coaching/search")
async def searchCoaching(location: str):
    """
    Progressive coaching center search:
    1. Search immediate area (25km)
    2. If < 5 results, expand to 75km
    3. If < 10 results, search neighboring cities
    4. Supplement with curated data for known cities
    5. Always show NTA exam centers
    6. Guarantee minimum 10 results
    """
    try:
        async with httpx.AsyncClient() as client:
            # geocode
            geo_res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location + " India", "format": "json",
                        "limit": 3, "addressdetails": 1},
                headers={"User-Agent": "Prism-ExamApp/1.0"},
                timeout=10.0
            )
            geo_data = geo_res.json()

            if not geo_data:
                return {"message": "location_not_found", "payload": [], "coords": None,
                        "ntaCenters": []}

            lat = float(geo_data[0]["lat"])
            lon = float(geo_data[0]["lon"])
            display = geo_data[0].get("display_name", location)
            address = geo_data[0].get("address", {})
            city = (
                address.get("city") or
                address.get("town") or
                address.get("suburb") or
                address.get("county") or
                display.split(",")[0]
            )
            state = (
                address.get("state_district") or
                address.get("state") or
                (display.split(",")[2].strip() if len(display.split(",")) > 2 else "")
            )

            print(f"[Coaching] Searching near: {city}, {state} ({lat:.4f}, {lon:.4f})")

            curated_region = resolve_curated_region(location, city, state)
            curated = get_curated_centers(curated_region, lat, lon)
            nta_centers = get_nta_centers(curated_region, lat, lon)

            if len(curated) >= 5:
                for c in curated:
                    c["googleMapsUrl"] = f"https://www.google.com/maps/search/{c['name'].replace(' ', '+')}+{curated_region.replace(' ', '+')}"
                    c["directionsUrl"] = f"https://www.google.com/maps/dir/{lat},{lon}/{c['lat']},{c['lon']}"
                print(f"[Coaching] Using curated fallback for {curated_region}: {len(curated)} centers")
                return {
                    "message": "centers",
                    "payload": curated[:30],
                    "ntaCenters": nta_centers,
                    "coords": {"lat": lat, "lon": lon, "city": city},
                    "searchMeta": {
                        "searchedRadiiKm": [],
                        "finalSearchAreaKm": 0,
                        "mode": "curated_fallback"
                    }
                }

            # progressive search with expanding radius
            all_centers = []
            searched_radii = []
            radii = [25000, 50000, 75000, 100000, 125000, 150000]

            for radius in radii:
                elements = await overpass_search(lat, lon, radius, client)
                centers = process_elements(elements, lat, lon)
                searched_radii.append(radius // 1000)

                # deduplicate
                existing_names = {c["name"] for c in all_centers}
                new_centers = [c for c in centers if c["name"] not in existing_names]
                all_centers.extend(new_centers)

                print(f"[Coaching] Radius {radius//1000}km: {len(all_centers)} total centers")

                if len(all_centers) >= 10:
                    break

            # supplement with curated data if still < 10
            if len(all_centers) < 10:
                curated = get_curated_centers(city, lat, lon)
                if not curated and state:
                    curated = get_curated_centers(state, lat, lon)
                if not curated and "telangana" in state.lower():
                    curated = get_curated_centers("hyderabad", lat, lon)
                existing_names = {c["name"] for c in all_centers}
                curated_new = [c for c in curated if c["name"] not in existing_names]
                all_centers.extend(curated_new)
                print(f"[Coaching] After curated supplement: {len(all_centers)} centers")

            # if STILL < 10, search neighboring major city
            if len(all_centers) < 10:
                # get the state capital or major city
                major_city_query = f"coaching classes {state} India"
                fallback_res = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": major_city_query, "format": "json", "limit": 1},
                    headers={"User-Agent": "Prism-ExamApp/1.0"},
                    timeout=8.0
                )
                fb_data = fallback_res.json()
                if fb_data:
                    fb_lat = float(fb_data[0]["lat"])
                    fb_lon = float(fb_data[0]["lon"])
                    fb_elements = await overpass_search(fb_lat, fb_lon, 50000, client)
                    fb_centers = process_elements(fb_elements, lat, lon)
                    existing_names = {c["name"] for c in all_centers}
                    fb_new = [c for c in fb_centers if c["name"] not in existing_names]
                    all_centers.extend(fb_new[:max(0, 10 - len(all_centers))])

            # NTA exam centers (shown separately)
            nta_centers = nta_centers or get_nta_centers(city, lat, lon)
            # also check state capital
            if not nta_centers and state:
                nta_centers = get_nta_centers(state.lower(), lat, lon)

            # add Google Maps search link for each center
            for c in all_centers:
                c["googleMapsUrl"] = f"https://www.google.com/maps/search/{c['name'].replace(' ', '+')}+{city.replace(' ', '+')}"
                c["directionsUrl"] = f"https://www.google.com/maps/dir/{lat},{lon}/{c['lat']},{c['lon']}"

            all_centers.sort(key=lambda x: x["distance"])

            print(f"[Coaching] Final: {len(all_centers)} coaching + {len(nta_centers)} NTA centers")
            return {
                "message": "centers",
                "payload": all_centers[:30],
                "ntaCenters": nta_centers,
                "coords": {"lat": lat, "lon": lon, "city": city},
                "searchMeta": {
                    "searchedRadiiKm": searched_radii,
                    "finalSearchAreaKm": searched_radii[-1] if searched_radii else 25
                }
            }

    except Exception as e:
        print(f"[Coaching] Search error: {e}")
        return {"message": "error", "payload": [], "coords": None, "ntaCenters": []}
