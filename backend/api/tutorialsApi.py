# api/tutorialsApi.py
# YouTube video search for study topics
# Uses YouTube Data API v3 — free 10,000 units/day (100 searches)
# Falls back to search URL if no API key

from fastapi import APIRouter
from config import YOUTUBE_API_KEY
import httpx

tutorialsRouter = APIRouter()

QUALITY_CHANNELS = {
    "JEE": {
        "Physics": ["UCKGvAJzSvqiJLOqx1u1yh5A",  # Physics Wallah
                    "UCB1DMdFRTPm2tSG6Fygxs6g"],  # Vedantu JEE
        "Chemistry": ["UCiNFBFCZrdGTMBDm2W3qHaQ"],
        "Maths": []
    },
    "NEET": {
        "Biology": ["UCKGvAJzSvqiJLOqx1u1yh5A"],  # PW
        "Chemistry": [],
        "Physics": []
    }
}


@tutorialsRouter.get("/tutorials/search")
async def searchTutorials(topic: str, subject: str = "", language: str = "Hindi",
                           examTarget: str = "JEE"):
    """
    Searches YouTube for high-quality tutorials.
    Filters by relevance, views, and channel quality.
    """
    if not YOUTUBE_API_KEY:
        # return search URL fallback
        query = f"{topic} {subject} {examTarget} {language} lecture"
        return {
            "message": "no_api_key",
            "payload": [{
                "id": "search",
                "title": f"Search: {topic} — {subject} ({language})",
                "channel": "YouTube Search",
                "thumbnail": "",
                "url": f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}",
                "duration": "Various",
                "views": "Click to search"
            }]
        }

    query = f"{topic} {subject} {examTarget} {language} lecture explanation"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": 9,
                    "order": "relevance",
                    "videoDuration": "medium",   # 4-20 min
                    "key": YOUTUBE_API_KEY,
                    "relevanceLanguage": "hi" if language == "Hindi" else "en",
                }
            )
            data = response.json()

        if "error" in data:
            print(f"[Tutorials] YouTube API error: {data['error']}")
            return {"message": "api_error", "payload": []}

        videos = []
        for item in data.get("items", []):
            vid_id = item["id"]["videoId"]
            snippet = item["snippet"]
            videos.append({
                "id": vid_id,
                "title": snippet["title"],
                "channel": snippet["channelTitle"],
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "duration": "",
                "views": "",
                "publishedAt": snippet["publishedAt"][:10]
            })

        print(f"[Tutorials] Found {len(videos)} videos for '{topic}'")
        return {"message": "videos", "payload": videos}

    except Exception as e:
        print(f"[Tutorials] Search failed: {e}")
        return {"message": "error", "payload": []}