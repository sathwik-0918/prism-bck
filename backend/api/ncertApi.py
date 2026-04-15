# api/ncertApi.py
# NCERT Line-by-Line reading feature
# Generates content from NCERT textbooks in RAG vector store
# Global caching — one user generates, all users benefit forever
# Supports Chemistry (JEE + NEET) and Biology (NEET only)
# Integrates mnemonics, tips, tricks from data folder

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict
from database.mongodb import get_db
from rag.nodes import main_llm, vector_store
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from config import GROQ_API_KEY
from datetime import datetime
import json, re, uuid

ncertRouter = APIRouter()

ncert_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=6000,
)

DEFAULT_NCERT_CHAPTERS = {
    ("Chemistry", "11"): [
        {"chapter": 1, "title": "Some Basic Concepts of Chemistry", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["mole concept", "stoichiometry", "atomic mass"], "pyqCount": 45, "difficulty": "medium"},
        {"chapter": 2, "title": "Structure of Atom", "weightage": "high", "examMarks": "3-6 marks", "tags": ["quantum numbers", "orbitals", "electronic configuration"], "pyqCount": 35, "difficulty": "medium"},
        {"chapter": 3, "title": "Classification of Elements and Periodicity in Properties", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["periodic table", "trends", "properties"], "pyqCount": 25, "difficulty": "easy"},
        {"chapter": 4, "title": "Chemical Bonding and Molecular Structure", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["VSEPR", "hybridization", "MOT"], "pyqCount": 50, "difficulty": "hard"},
        {"chapter": 5, "title": "States of Matter", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["gas laws", "kinetic theory", "liquids"], "pyqCount": 20, "difficulty": "medium"},
        {"chapter": 6, "title": "Thermodynamics", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["enthalpy", "first law", "Hess law"], "pyqCount": 42, "difficulty": "hard"},
        {"chapter": 7, "title": "Equilibrium", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["ionic equilibrium", "Le Chatelier", "pH"], "pyqCount": 48, "difficulty": "hard"},
        {"chapter": 8, "title": "Redox Reactions", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["oxidation number", "balancing", "disproportionation"], "pyqCount": 18, "difficulty": "medium"},
        {"chapter": 9, "title": "Hydrogen", "weightage": "low", "examMarks": "1-2 marks", "tags": ["hydrides", "water", "heavy water"], "pyqCount": 12, "difficulty": "easy"},
        {"chapter": 10, "title": "The s-Block Elements", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["alkali metals", "alkaline earth metals", "anomalous behavior"], "pyqCount": 22, "difficulty": "medium"},
        {"chapter": 11, "title": "The p-Block Elements", "weightage": "high", "examMarks": "3-6 marks", "tags": ["boron family", "carbon family", "nitrogen family"], "pyqCount": 34, "difficulty": "hard"},
        {"chapter": 12, "title": "Organic Chemistry - Some Basic Principles and Techniques", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["IUPAC", "isomerism", "electronic effects"], "pyqCount": 46, "difficulty": "hard"},
        {"chapter": 13, "title": "Hydrocarbons", "weightage": "high", "examMarks": "3-6 marks", "tags": ["alkanes", "alkenes", "aromatic hydrocarbons"], "pyqCount": 32, "difficulty": "medium"},
        {"chapter": 14, "title": "Environmental Chemistry", "weightage": "low", "examMarks": "1-2 marks", "tags": ["pollution", "green chemistry", "atmosphere"], "pyqCount": 10, "difficulty": "easy"}
    ],
    ("Chemistry", "12"): [
        {"chapter": 1, "title": "The Solid State", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["unit cell", "packing", "defects"], "pyqCount": 20, "difficulty": "medium"},
        {"chapter": 2, "title": "Solutions", "weightage": "high", "examMarks": "3-5 marks", "tags": ["colligative properties", "Raoult's law", "osmotic pressure"], "pyqCount": 28, "difficulty": "medium"},
        {"chapter": 3, "title": "Electrochemistry", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["Nernst equation", "conductance", "cells"], "pyqCount": 40, "difficulty": "hard"},
        {"chapter": 4, "title": "Chemical Kinetics", "weightage": "high", "examMarks": "3-5 marks", "tags": ["rate law", "order", "half life"], "pyqCount": 30, "difficulty": "medium"},
        {"chapter": 5, "title": "Surface Chemistry", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["adsorption", "colloids", "catalysis"], "pyqCount": 18, "difficulty": "easy"},
        {"chapter": 6, "title": "General Principles and Processes of Isolation of Elements", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["metallurgy", "concentration", "electrolytic reduction"], "pyqCount": 16, "difficulty": "medium"},
        {"chapter": 7, "title": "The p-Block Elements", "weightage": "high", "examMarks": "3-6 marks", "tags": ["group 15", "group 16", "group 17"], "pyqCount": 26, "difficulty": "hard"},
        {"chapter": 8, "title": "The d- and f-Block Elements", "weightage": "high", "examMarks": "3-6 marks", "tags": ["transition elements", "lanthanoids", "actinoids"], "pyqCount": 29, "difficulty": "medium"},
        {"chapter": 9, "title": "Coordination Compounds", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["Werner theory", "isomerism", "VBT"], "pyqCount": 38, "difficulty": "hard"},
        {"chapter": 10, "title": "Haloalkanes and Haloarenes", "weightage": "high", "examMarks": "3-5 marks", "tags": ["SN1", "SN2", "reactions"], "pyqCount": 30, "difficulty": "medium"},
        {"chapter": 11, "title": "Alcohols, Phenols and Ethers", "weightage": "high", "examMarks": "3-5 marks", "tags": ["acidity", "reactions", "preparation"], "pyqCount": 31, "difficulty": "medium"},
        {"chapter": 12, "title": "Aldehydes, Ketones and Carboxylic Acids", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["nucleophilic addition", "named reactions", "acidity"], "pyqCount": 44, "difficulty": "hard"},
        {"chapter": 13, "title": "Amines", "weightage": "high", "examMarks": "3-5 marks", "tags": ["basicity", "diazotization", "carbylamine"], "pyqCount": 27, "difficulty": "medium"},
        {"chapter": 14, "title": "Biomolecules", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["carbohydrates", "proteins", "DNA"], "pyqCount": 19, "difficulty": "easy"},
        {"chapter": 15, "title": "Polymers", "weightage": "low", "examMarks": "1-2 marks", "tags": ["addition polymerization", "condensation", "copolymers"], "pyqCount": 12, "difficulty": "easy"},
        {"chapter": 16, "title": "Chemistry in Everyday Life", "weightage": "low", "examMarks": "1-2 marks", "tags": ["drugs", "soaps", "detergents"], "pyqCount": 10, "difficulty": "easy"}
    ],
    ("Biology", "11"): [
        {"chapter": 1, "title": "The Living World", "weightage": "medium", "examMarks": "1-3 marks", "tags": ["taxonomy", "classification", "biodiversity"], "pyqCount": 18, "difficulty": "easy"},
        {"chapter": 2, "title": "Biological Classification", "weightage": "high", "examMarks": "2-4 marks", "tags": ["five kingdom", "monera", "protista"], "pyqCount": 25, "difficulty": "medium"},
        {"chapter": 3, "title": "Plant Kingdom", "weightage": "high", "examMarks": "2-4 marks", "tags": ["algae", "bryophytes", "pteridophytes"], "pyqCount": 24, "difficulty": "medium"},
        {"chapter": 4, "title": "Animal Kingdom", "weightage": "very_high", "examMarks": "4-6 marks", "tags": ["classification", "chordates", "non-chordates"], "pyqCount": 42, "difficulty": "hard"},
        {"chapter": 5, "title": "Morphology of Flowering Plants", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["root", "stem", "leaf"], "pyqCount": 20, "difficulty": "medium"},
        {"chapter": 6, "title": "Anatomy of Flowering Plants", "weightage": "high", "examMarks": "3-5 marks", "tags": ["tissues", "xylem", "stomata"], "pyqCount": 26, "difficulty": "medium"},
        {"chapter": 7, "title": "Structural Organisation in Animals", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["tissues", "cockroach", "earthworm"], "pyqCount": 18, "difficulty": "medium"},
        {"chapter": 8, "title": "Cell: The Unit of Life", "weightage": "very_high", "examMarks": "4-6 marks", "tags": ["cell organelles", "membrane", "cell theory"], "pyqCount": 40, "difficulty": "medium"},
        {"chapter": 9, "title": "Biomolecules", "weightage": "high", "examMarks": "3-5 marks", "tags": ["proteins", "carbohydrates", "enzymes"], "pyqCount": 28, "difficulty": "medium"},
        {"chapter": 10, "title": "Cell Cycle and Cell Division", "weightage": "very_high", "examMarks": "4-6 marks", "tags": ["mitosis", "meiosis", "checkpoints"], "pyqCount": 34, "difficulty": "hard"},
        {"chapter": 11, "title": "Transport in Plants", "weightage": "high", "examMarks": "3-5 marks", "tags": ["xylem", "phloem", "transpiration"], "pyqCount": 27, "difficulty": "medium"},
        {"chapter": 12, "title": "Mineral Nutrition", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["essential elements", "deficiency", "nitrogen cycle"], "pyqCount": 17, "difficulty": "medium"},
        {"chapter": 13, "title": "Photosynthesis in Higher Plants", "weightage": "very_high", "examMarks": "4-6 marks", "tags": ["light reaction", "Calvin cycle", "C4 pathway"], "pyqCount": 35, "difficulty": "hard"},
        {"chapter": 14, "title": "Respiration in Plants", "weightage": "high", "examMarks": "3-5 marks", "tags": ["glycolysis", "Krebs cycle", "ETS"], "pyqCount": 30, "difficulty": "hard"},
        {"chapter": 15, "title": "Plant Growth and Development", "weightage": "high", "examMarks": "3-5 marks", "tags": ["hormones", "photoperiodism", "vernalization"], "pyqCount": 28, "difficulty": "medium"},
        {"chapter": 16, "title": "Digestion and Absorption", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["enzymes", "small intestine", "absorption"], "pyqCount": 18, "difficulty": "medium"},
        {"chapter": 17, "title": "Breathing and Exchange of Gases", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["respiration", "oxygen dissociation", "transport"], "pyqCount": 19, "difficulty": "medium"},
        {"chapter": 18, "title": "Body Fluids and Circulation", "weightage": "high", "examMarks": "3-5 marks", "tags": ["blood", "cardiac cycle", "lymph"], "pyqCount": 24, "difficulty": "medium"},
        {"chapter": 19, "title": "Excretory Products and their Elimination", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["nephron", "urine formation", "osmoregulation"], "pyqCount": 20, "difficulty": "medium"},
        {"chapter": 20, "title": "Locomotion and Movement", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["muscle contraction", "joints", "skeletal system"], "pyqCount": 16, "difficulty": "easy"},
        {"chapter": 21, "title": "Neural Control and Coordination", "weightage": "high", "examMarks": "3-5 marks", "tags": ["neuron", "synapse", "brain"], "pyqCount": 23, "difficulty": "medium"},
        {"chapter": 22, "title": "Chemical Coordination and Integration", "weightage": "high", "examMarks": "3-5 marks", "tags": ["hormones", "endocrine glands", "feedback"], "pyqCount": 25, "difficulty": "medium"}
    ],
    ("Biology", "12"): [
        {"chapter": 1, "title": "Reproduction in Organisms", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["asexual reproduction", "life span", "reproductive health"], "pyqCount": 18, "difficulty": "easy"},
        {"chapter": 2, "title": "Sexual Reproduction in Flowering Plants", "weightage": "high", "examMarks": "3-5 marks", "tags": ["pollination", "embryo sac", "double fertilization"], "pyqCount": 30, "difficulty": "medium"},
        {"chapter": 3, "title": "Human Reproduction", "weightage": "high", "examMarks": "3-5 marks", "tags": ["gametogenesis", "menstrual cycle", "fertilization"], "pyqCount": 26, "difficulty": "medium"},
        {"chapter": 4, "title": "Reproductive Health", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["contraception", "ART", "STDs"], "pyqCount": 18, "difficulty": "easy"},
        {"chapter": 5, "title": "Principles of Inheritance and Variation", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["Mendel", "genetics", "chromosomes"], "pyqCount": 45, "difficulty": "hard"},
        {"chapter": 6, "title": "Molecular Basis of Inheritance", "weightage": "very_high", "examMarks": "4-8 marks", "tags": ["DNA replication", "transcription", "genetic code"], "pyqCount": 44, "difficulty": "hard"},
        {"chapter": 7, "title": "Evolution", "weightage": "high", "examMarks": "3-5 marks", "tags": ["Hardy-Weinberg", "speciation", "human evolution"], "pyqCount": 28, "difficulty": "medium"},
        {"chapter": 8, "title": "Human Health and Disease", "weightage": "very_high", "examMarks": "4-6 marks", "tags": ["immunity", "AIDS", "cancer"], "pyqCount": 38, "difficulty": "medium"},
        {"chapter": 9, "title": "Strategies for Enhancement in Food Production", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["plant breeding", "animal husbandry", "single cell protein"], "pyqCount": 20, "difficulty": "easy"},
        {"chapter": 10, "title": "Microbes in Human Welfare", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["biogas", "antibiotics", "sewage treatment"], "pyqCount": 19, "difficulty": "easy"},
        {"chapter": 11, "title": "Biotechnology: Principles and Processes", "weightage": "high", "examMarks": "3-5 marks", "tags": ["restriction enzymes", "vectors", "PCR"], "pyqCount": 27, "difficulty": "medium"},
        {"chapter": 12, "title": "Biotechnology and its Applications", "weightage": "high", "examMarks": "3-5 marks", "tags": ["Bt cotton", "gene therapy", "biopatent"], "pyqCount": 25, "difficulty": "medium"},
        {"chapter": 13, "title": "Organisms and Populations", "weightage": "high", "examMarks": "3-5 marks", "tags": ["population", "ecology", "adaptation"], "pyqCount": 26, "difficulty": "medium"},
        {"chapter": 14, "title": "Ecosystem", "weightage": "high", "examMarks": "3-5 marks", "tags": ["energy flow", "succession", "productivity"], "pyqCount": 24, "difficulty": "medium"},
        {"chapter": 15, "title": "Biodiversity and Conservation", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["hotspots", "in situ", "ex situ"], "pyqCount": 18, "difficulty": "easy"},
        {"chapter": 16, "title": "Environmental Issues", "weightage": "medium", "examMarks": "2-4 marks", "tags": ["pollution", "global warming", "ozone depletion"], "pyqCount": 17, "difficulty": "easy"}
    ]
}


def fallback_ncert_chapters(subject: str, class_num: str) -> list:
    return DEFAULT_NCERT_CHAPTERS.get((subject, class_num), [])


def merge_with_default_chapters(subject: str, class_num: str, chapters: list) -> list:
    """
    Guarantees a complete per-class catalog even if RAG/LLM returns only a partial list.
    Preserves richer metadata from discovered chapters when chapter numbers match.
    """
    defaults = fallback_ncert_chapters(subject, class_num)
    if not defaults:
        return chapters or []

    chapters_by_number = {
        int(ch.get("chapter")): ch
        for ch in (chapters or [])
        if isinstance(ch, dict) and str(ch.get("chapter", "")).isdigit()
    }

    merged = []
    for default_chapter in defaults:
        chapter_num = int(default_chapter["chapter"])
        discovered = chapters_by_number.get(chapter_num, {})
        merged.append({
            **default_chapter,
            **discovered,
            "chapter": chapter_num
        })

    return merged


def normalize_topic_label(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    return value


def slugify_topic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def build_default_subtopics(chapter_info: dict) -> list:
    title = normalize_topic_label(chapter_info.get("title", "") or "Overview")
    tags = [normalize_topic_label(tag) for tag in (chapter_info.get("tags") or []) if normalize_topic_label(tag)]

    defaults = []
    if title:
        defaults.append({
            "subtopic": f"{title} Overview",
            "isKeyTopic": True,
            "estimatedReadTime": "6-8 min",
            "examImportance": "high"
        })

    for idx, tag in enumerate(tags):
        defaults.append({
            "subtopic": tag.title(),
            "isKeyTopic": idx < 2,
            "estimatedReadTime": "5-7 min",
            "examImportance": "high" if idx < 2 else "medium"
        })

    if title and "chemistry" in title.lower():
        defaults.append({
            "subtopic": "Important Reactions and Exceptions",
            "isKeyTopic": True,
            "estimatedReadTime": "6-8 min",
            "examImportance": "high"
        })

    return defaults


def merge_with_default_subtopics(chapter_info: dict, subtopics: list) -> list:
    defaults = build_default_subtopics(chapter_info)
    merged = []
    seen = set()

    for item in (subtopics or []) + defaults:
        if isinstance(item, str):
            item = {"subtopic": item}
        if not isinstance(item, dict):
            continue

        name = normalize_topic_label(item.get("subtopic", ""))
        if not name:
            continue

        key = slugify_topic(name)
        if key in seen:
            continue
        seen.add(key)

        merged.append({
            "subtopic": name,
            "isKeyTopic": bool(item.get("isKeyTopic", False)),
            "estimatedReadTime": item.get("estimatedReadTime", "5-8 min"),
            "examImportance": item.get("examImportance", "medium")
        })

    return merged


def has_sufficient_subtopics(subtopics: list, chapter_info: dict) -> bool:
    tags = chapter_info.get("tags") or []
    minimum_expected = max(4, min(8, len(tags) + 2))
    return len(subtopics or []) >= minimum_expected


def tokenize_for_match(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", (value or "").lower())
        if len(token) > 2
    }


def sanitize_mnemonic_text(mnemonic: str, mnemonic_context: str, topic: str, chapter_title: str) -> str:
    mnemonic = normalize_topic_label(mnemonic)
    if not mnemonic or not mnemonic_context.strip():
        return ""

    bad_prefixes = {
        "memory trick if available",
        "not available",
        "no mnemonic",
        "none",
        "n/a"
    }
    if mnemonic.lower() in bad_prefixes:
        return ""

    topic_tokens = tokenize_for_match(topic) | tokenize_for_match(chapter_title)
    context_tokens = tokenize_for_match(mnemonic_context)
    mnemonic_tokens = tokenize_for_match(mnemonic)

    if topic_tokens and not (topic_tokens & context_tokens):
        return ""
    if mnemonic_tokens and topic_tokens and not (mnemonic_tokens & topic_tokens) and len(mnemonic.split()) < 3:
        return ""

    return mnemonic[:240]


def is_low_quality_generated_content(content: dict | None) -> bool:
    if not content:
        return True

    sections = content.get("sections") or []
    if len(sections) < 3:
        return True

    section_text = " ".join((section.get("content") or "").strip().lower() for section in sections)
    template_markers = [
        "clear paragraph explanation from ncert perspective",
        "detailed explanation with all ncert points covered line by line",
        "all reactions with mechanisms mentioned in ncert",
        "properties table and uses as given in ncert"
    ]
    return any(marker in section_text for marker in template_markers)


def parse_topic_cache_key(cache_key: str) -> dict | None:
    parts = (cache_key or "").split("_", 3)
    if len(parts) != 4:
        return None

    subject, class_num, chapter_num, topic_slug = parts
    if not chapter_num.isdigit():
        return None

    return {
        "subject": subject.title(),
        "classNum": class_num,
        "chapterNum": int(chapter_num),
        "topic": topic_slug.replace("_", " ").strip()
    }


def now():
    return datetime.utcnow().isoformat() + "Z"


def is_placeholder_content(content: dict | None) -> bool:
    """Detect old fallback payloads that should not be reused forever."""
    if not content:
        return False
    sections = content.get("sections") or []
    if not sections:
        return False
    first_heading = (sections[0].get("heading") or "").strip().lower()
    return first_heading == "content loading"


async def discover_chapters_from_rag(subject: str, class_num: str) -> list:
    """
    Uses RAG to discover actual chapters from NCERT textbooks.
    Queries vector store for chapter info from the actual PDF content.
    """
    search_queries = [
        f"NCERT {subject} Class {class_num} chapter list contents",
        f"{subject} Class {class_num} textbook chapters topics",
        f"NCERT {class_num} {subject} unit chapter names"
    ]

    all_text = []
    for query in search_queries:
        results = vector_store.query(query_text=query, top_k=8)
        for r in results:
            meta = r.get("metadata", {})
            source = meta.get("source", "").lower()
            text = meta.get("text", "")
            if text and any(kw in source for kw in
                            ["ncert", "part", "class", "11", "12", "bio", "chem",
                             "phy", "textbook", "cbse"]):
                all_text.append(text)

    context = "\n\n".join(all_text[:10])[:4000]

    system_prompt = f"""You are analyzing NCERT {subject} Class {class_num} textbook content.
From the provided textbook content, extract the actual chapter list.
Return ONLY valid JSON array — no backslashes in values:
[
  {{
    "chapter": 1,
    "title": "Exact chapter title from textbook",
    "weightage": "very_high|high|medium|low",
    "examMarks": "4-8 marks",
    "tags": ["key topic 1", "key topic 2", "key topic 3"],
    "pyqCount": 45,
    "difficulty": "hard|medium|easy"
  }}
]

Base weightage on actual JEE/NEET exam importance.
Return chapters in order as they appear in the textbook.
If content is limited, generate the standard NCERT {subject} Class {class_num} chapters."""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Textbook content:\n{context}\n\nGenerate chapter list for NCERT {subject} Class {class_num}")
        ])

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            chapters = json.loads(match.group())
            return chapters
    except Exception as e:
        print(f"[NCERT] Chapter discovery error: {e}")

    return []


async def discover_subtopics_from_rag(
    subject: str, class_num: str, chapter_title: str
) -> list:
    """
    Discovers actual subtopics within a chapter from NCERT content.
    Uses RAG to find what topics are actually covered.
    """
    search_queries = [
        f"NCERT {subject} {chapter_title} topics subtopics sections",
        f"{subject} {chapter_title} headings subheadings ncert",
        f"{chapter_title} important concepts class {class_num} ncert"
    ]

    chunks = []
    for query in search_queries:
        results = vector_store.query(query_text=query, top_k=10)
        for result in results:
            metadata = result.get("metadata") or {}
            text = metadata.get("text", "")
            source = metadata.get("source", "").lower()
            if text and ("ncert" in source or "class" in source or subject.lower() in source):
                chunks.append(text)

    context = "\n\n".join(chunks[:20])[:8000]

    system_prompt = f"""Analyze NCERT {subject} Class {class_num} - Chapter: {chapter_title}
Extract the exact subtopics/sections covered in this chapter.
Return ONLY valid JSON array:
[
  {{
    "subtopic": "Exact subtopic name from NCERT",
    "isKeyTopic": true,
    "estimatedReadTime": "5-8 min",
    "examImportance": "high|medium|low"
  }}
]
Include all important subtopics that appear in NCERT. Order by textbook flow and return at least 6 subtopics whenever the chapter has enough material."""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chapter content:\n{context}")
        ])

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[NCERT] Subtopic discovery error: {e}")

    return []

@ncertRouter.get("/ncert/discover/{subject}/{classNum}")
async def discoverChapters(subject: str, classNum: str):
    """
    RAG-driven chapter discovery.
    Cached globally — generated once, used by all users.
    """
    db = get_db()
    cache_key = f"chapters_{subject}_{classNum}"
    default_chapters = fallback_ncert_chapters(subject, classNum)

    cached = await db.ncert_chapters.find_one({"cacheKey": cache_key}, {"_id": 0})
    if cached:
        merged_chapters = merge_with_default_chapters(subject, classNum, cached.get("chapters", []))
        cache_needs_refresh = len(merged_chapters) != len(cached.get("chapters", []))
        if cache_needs_refresh:
            await db.ncert_chapters.update_one(
                {"cacheKey": cache_key},
                {"$set": {"chapters": merged_chapters}}
            )
        print(f"[NCERT] Chapters cache HIT: {cache_key}")
        chapters = merged_chapters
        visible_count = max(1, min(int(cached.get("visibleChapterCount", 1)), len(chapters) or 1))
        return {
            "message": "chapters",
            "payload": chapters,
            "visibleChapterCount": visible_count,
            "cached": True
        }

    print(f"[NCERT] Discovering chapters for {subject} Class {classNum}...")
    chapters = await discover_chapters_from_rag(subject, classNum)
    chapters = merge_with_default_chapters(subject, classNum, chapters)
    if not chapters:
        print(f"[NCERT] Using fallback chapter catalog for {subject} Class {classNum}")
        chapters = default_chapters

    if chapters:
        await db.ncert_chapters.update_one(
            {"cacheKey": cache_key},
            {"$set": {
                "cacheKey": cache_key,
                "subject": subject,
                "classNum": classNum,
                "chapters": chapters,
                "visibleChapterCount": 1,
                "generatedAt": now()
            }},
            upsert=True
        )

    return {
        "message": "chapters",
        "payload": chapters,
        "visibleChapterCount": 1,
        "cached": False
    }


class RevealChapterRequest(BaseModel):
    subject: str
    classNum: str


@ncertRouter.post("/ncert/reveal-next")
async def revealNextChapter(req: RevealChapterRequest):
    """
    Advances globally visible chapter count for a subject/class.
    Once one user reveals the next chapter, everyone sees it.
    """
    db = get_db()
    cache_key = f"chapters_{req.subject}_{req.classNum}"
    cached = await db.ncert_chapters.find_one({"cacheKey": cache_key}, {"_id": 0, "chapters": 1, "visibleChapterCount": 1})

    chapters = (cached or {}).get("chapters", [])
    if not chapters:
        return {"message": "chapters not found", "payload": None}

    current_visible = max(1, min(int((cached or {}).get("visibleChapterCount", 1)), len(chapters)))
    next_visible = min(current_visible + 1, len(chapters))

    await db.ncert_chapters.update_one(
        {"cacheKey": cache_key},
        {"$set": {"visibleChapterCount": next_visible}}
    )

    return {
        "message": "chapter revealed",
        "payload": {
            "visibleChapterCount": next_visible,
            "totalChapters": len(chapters)
        }
    }


@ncertRouter.get("/ncert/subtopics/{subject}/{classNum}/{chapterNum}")
async def discoverSubtopics(subject: str, classNum: str, chapterNum: int, chapterTitle: str = ""):
    """
    RAG-driven subtopic discovery for a chapter.
    Cached globally.
    """
    db = get_db()
    cache_key = f"subtopics_{subject}_{classNum}_{chapterNum}"
    chapters_cache = await db.ncert_chapters.find_one(
        {"cacheKey": f"chapters_{subject}_{classNum}"},
        {"_id": 0, "chapters": 1}
    )
    chapters = (chapters_cache or {}).get("chapters", [])
    chapter_info = next((c for c in chapters if c.get("chapter") == chapterNum), None) or {
        "chapter": chapterNum,
        "title": chapterTitle or f"Chapter {chapterNum}",
        "tags": []
    }

    cached = await db.ncert_subtopics.find_one({"cacheKey": cache_key}, {"_id": 0})
    if cached:
        merged_subtopics = merge_with_default_subtopics(chapter_info, cached.get("subtopics", []))
        cache_needs_refresh = (
            len(merged_subtopics) != len(cached.get("subtopics", []))
            or not has_sufficient_subtopics(cached.get("subtopics", []), chapter_info)
        )
        if cache_needs_refresh:
            refreshed_subtopics = await discover_subtopics_from_rag(subject, classNum, chapter_info.get("title", chapterTitle or ""))
            merged_subtopics = merge_with_default_subtopics(chapter_info, refreshed_subtopics)
            await db.ncert_subtopics.update_one(
                {"cacheKey": cache_key},
                {"$set": {
                    "subtopics": merged_subtopics,
                    "chapterTitle": chapter_info.get("title", chapterTitle or ""),
                    "visibleTopicCount": min(
                        max(1, int(cached.get("visibleTopicCount", 1))),
                        len(merged_subtopics) or 1
                    ),
                    "generatedAt": now()
                }},
                upsert=True
            )
            cached["visibleTopicCount"] = min(
                max(1, int(cached.get("visibleTopicCount", 1))),
                len(merged_subtopics) or 1
            )
        return {
            "message": "subtopics",
            "payload": merged_subtopics,
            "visibleTopicCount": min(
                max(1, int(cached.get("visibleTopicCount", 1))),
                len(merged_subtopics) or 1
            ),
            "cached": True
        }

    print(f"[NCERT] Discovering subtopics: {chapterTitle}...")
    subtopics = await discover_subtopics_from_rag(subject, classNum, chapter_info.get("title", chapterTitle or ""))
    subtopics = merge_with_default_subtopics(chapter_info, subtopics)

    if subtopics:
        await db.ncert_subtopics.update_one(
            {"cacheKey": cache_key},
            {"$set": {
                "cacheKey": cache_key,
                "subject": subject,
                "classNum": classNum,
                "chapterNum": chapterNum,
                "chapterTitle": chapter_info.get("title", chapterTitle or ""),
                "subtopics": subtopics,
                "visibleTopicCount": 1,
                "generatedAt": now()
            }},
            upsert=True
        )

    return {
        "message": "subtopics",
        "payload": subtopics,
        "visibleTopicCount": 1,
        "cached": False
    }


class RevealTopicRequest(BaseModel):
    subject: str
    classNum: str
    chapterNum: int


@ncertRouter.post("/ncert/reveal-next-topic")
async def revealNextTopic(req: RevealTopicRequest):
    db = get_db()
    cache_key = f"subtopics_{req.subject}_{req.classNum}_{req.chapterNum}"
    cached = await db.ncert_subtopics.find_one(
        {"cacheKey": cache_key},
        {"_id": 0, "subtopics": 1, "visibleTopicCount": 1}
    )

    subtopics = (cached or {}).get("subtopics", [])
    if not subtopics:
        return {"message": "subtopics not found", "payload": None}

    current_visible = max(1, min(int((cached or {}).get("visibleTopicCount", 1)), len(subtopics)))
    next_visible = min(current_visible + 1, len(subtopics))

    await db.ncert_subtopics.update_one(
        {"cacheKey": cache_key},
        {"$set": {"visibleTopicCount": next_visible}}
    )

    return {
        "message": "topic revealed",
        "payload": {
            "visibleTopicCount": next_visible,
            "totalTopics": len(subtopics)
        }
    }


@ncertRouter.post("/ncert/regenerate")
async def regenerateContent(req: "GenerateRequest"):
    """
    Forces regeneration of content (ignores cache).
    Used when user clicks Regenerate button.
    """
    db = get_db()
    class_key = f"Class {req.classNum}"
    cache_key = f"{req.subject}_{req.classNum}_{req.chapterNum}_{req.topic}".replace(" ", "_").lower()

    await db.ncert_content.delete_one({"cacheKey": cache_key})

    chapters_cache = await db.ncert_chapters.find_one({
        "cacheKey": f"chapters_{req.subject}_{req.classNum}"
    })
    chapter_info = {}
    if chapters_cache:
        chapters = chapters_cache.get("chapters", [])
        chapter_info = next((c for c in chapters if c.get("chapter") == req.chapterNum), {})

    print(f"[NCERT] Regenerating: {req.subject} {req.classNum} Ch{req.chapterNum} - {req.topic}")
    content = await generate_topic_content(
        req.subject, class_key, chapter_info or {"title": req.topic}, req.topic, db
    )

    content["cacheKey"] = cache_key
    content["generatedAt"] = now()

    await db.ncert_content.insert_one(content)
    content.pop("_id", None)

    return {"message": "content", "payload": content, "cached": False}

# NCERT source file patterns in our vector store
NCERT_SOURCES = {
    "Chemistry": [
        "chem", "chemistry", "ncert_chem", "11_chem", "12_chem",
        "organic", "inorganic", "physical_chem"
    ],
    "Biology": [
        "bio", "biology", "ncert_bio", "11_bio", "12_bio",
        "botany", "zoology"
    ]
}


async def query_ncert_content(topic: str, subject: str, chapter_title: str, db) -> str:
    """
    Queries vector store filtered to NCERT sources only.
    Returns combined relevant text chunks.
    """
    search_queries = [
        f"{chapter_title} {topic}",
        f"NCERT {subject} {chapter_title}",
        f"{topic} {subject} class 11 12 NCERT",
        f"{subject} {chapter_title} important concepts",
        f"{subject} {topic} examples exceptions ncert",
        f"{subject} {chapter_title} line by line"
    ]

    all_texts = []
    seen_texts = set()

    for query in search_queries:
        # Wider retrieval for NCERT "line-by-line" depth.
        results = vector_store.query(query_text=query, top_k=20)
        for r in results:
            meta = r.get("metadata", {})
            source = meta.get("source", "").lower()
            text = meta.get("text", "")

            # filter to likely NCERT sources
            source_keywords = NCERT_SOURCES.get(subject, [])
            is_ncert = (
                any(kw in source for kw in source_keywords) or
                "ncert" in source or
                "part" in source or
                "class" in source or
                any(kw in source for kw in ["11", "12", "textbook"])
            )

            if text and is_ncert and text not in seen_texts:
                seen_texts.add(text)
                all_texts.append(text)

    # Preserve much larger NCERT coverage (not only first 10 chunks).
    return "\n\n".join(all_texts[:120])


async def generate_topic_content(
    subject: str,
    class_name: str,
    chapter_info: dict,
    topic: str,
    db
) -> dict:
    """
    Generates comprehensive content for a single topic within a chapter.
    Includes: explanation, key points, mnemonics, examples, diagram info.
    Uses NCERT sources from vector store + mnemonic data folder.
    """
    chapter_title = chapter_info["title"]

    # get NCERT content
    ncert_context = await query_ncert_content(topic, subject, chapter_title, db)

    # also search for mnemonics/tips
    mnemonic_results = vector_store.query(
        query_text=f"mnemonic trick tip {topic} {chapter_title} {subject} remember",
        top_k=8
    )
    topic_tokens = tokenize_for_match(topic) | tokenize_for_match(chapter_title)
    mnemonic_chunks = []
    for result in mnemonic_results:
        metadata = result.get("metadata") or {}
        text = metadata.get("text", "")
        source = metadata.get("source", "").lower()
        source_is_mnemonic = any(keyword in source for keyword in ["mnemonic", "trick", "tip", "memory"])
        text_tokens = tokenize_for_match(text)
        has_topic_match = not topic_tokens or bool(topic_tokens & text_tokens)
        if text and source_is_mnemonic and has_topic_match:
            mnemonic_chunks.append(text)

    mnemonic_context = "\n\n".join(mnemonic_chunks[:8])
    mnemonic_available = bool(mnemonic_context.strip())

    system_prompt = f"""You are an expert {subject} teacher preparing {class_name} NCERT content for JEE/NEET.
Generate comprehensive educational content for the topic.

Return ONLY valid JSON (no LaTeX backslashes in JSON values, use plain text for formulas):
{{
  "topic": "{topic}",
  "chapterTitle": "{chapter_title}",
  "subject": "{subject}",
  "class": "{class_name}",
  "readTime": "5-10 min",
  "sections": [
    {{
      "heading": "Introduction / What is it?",
      "content": "Clear paragraph explanation from NCERT perspective",
      "isKey": false
    }},
    {{
      "heading": "Key Concepts",
      "content": "Detailed explanation with all NCERT points covered line by line",
      "isKey": true
    }},
    {{
      "heading": "Important Reactions / Processes",
      "content": "All reactions with mechanisms mentioned in NCERT",
      "isKey": true
    }},
    {{
      "heading": "Properties and Applications",
      "content": "Properties table and uses as given in NCERT",
      "isKey": false
    }}
  ],
  "keyPoints": [
    "Exact NCERT important point 1",
    "Exact NCERT important point 2"
  ],
  "formulas": [
    {{"name": "Formula name", "formula": "plain text formula", "unit": "unit if any"}}
  ],
  "mnemonic": "Memory trick if available",
  "funFact": "Interesting scenario or joke to make this memorable",
  "diagrams": [
    {{
      "title": "Diagram title as in NCERT",
      "description": "What the diagram shows",
      "searchQuery": "exact Google Images search query to find this diagram"
    }}
  ],
  "examTip": "What specifically to note for JEE/NEET exam",
  "ncertExactLines": [
    "Important line from NCERT that must be memorized exactly"
  ],
  "commonErrors": ["Common mistake students make"]
}}"""

    user_message = f"""Generate NCERT line-by-line content for:
Subject: {subject} | Class: {class_name}
Chapter: {chapter_title} | Topic: {topic}

NCERT Source Content:
{ncert_context[:50000]}

Mnemonic/Tips Available:
{mnemonic_context[:8000]}

Cover ALL points from NCERT thoroughly. Do not skip minor exceptions, notes, or examples that can appear in exams.
If content is long, still prioritize completeness over brevity and preserve exam-critical details.
Only include a mnemonic if one is explicitly supported by the Mnemonic/Tips context. Otherwise return an empty string for mnemonic."""

    try:
        response = ncert_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # fix backslash escape issue
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            json_str = match.group()
            # escape unescaped backslashes
            json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            parsed = json.loads(json_str)
            if is_low_quality_generated_content(parsed):
                raise ValueError("Low quality templated content")
            parsed["mnemonic"] = sanitize_mnemonic_text(
                parsed.get("mnemonic", ""),
                mnemonic_context,
                topic,
                chapter_title
            )
            parsed["mnemonicSourceAvailable"] = mnemonic_available and bool(parsed["mnemonic"])
            return parsed
    except Exception as e:
        print(f"[NCERT] Generation error for {topic}: {e}")

    # deterministic fallback for rate limits/JSON errors:
    # use retrieved NCERT chunks so user still gets useful content.
    fallback_lines = [ln.strip() for ln in ncert_context.split("\n") if ln.strip()]
    short_context = " ".join(fallback_lines)[:12000]
    para_chunks = [p.strip() for p in short_context.split(". ") if p.strip()]
    intro = ". ".join(para_chunks[:3]).strip()
    key_concepts = ". ".join(para_chunks[3:8]).strip()
    reactions = ". ".join(para_chunks[8:12]).strip()

    if intro and not intro.endswith("."):
        intro += "."
    if key_concepts and not key_concepts.endswith("."):
        key_concepts += "."
    if reactions and not reactions.endswith("."):
        reactions += "."

    key_points = []
    for sentence in para_chunks[:6]:
        s = sentence.strip()
        if len(s) > 20:
            key_points.append(s if s.endswith(".") else f"{s}.")
    key_points = key_points[:5]

    mnemonic_hint = ""
    if mnemonic_context.strip():
        mnemonic_hint = sanitize_mnemonic_text(
            mnemonic_context.strip().split("\n")[0][:180],
            mnemonic_context,
            topic,
            chapter_title
        )

    # fallback
    return {
        "topic": topic,
        "chapterTitle": chapter_title,
        "subject": subject,
        "class": class_name,
        "readTime": "10-15 min",
        "sections": [
            {
                "heading": "Introduction / What is it?",
                "content": intro or f"{topic} is an important part of {chapter_title}.",
                "isKey": True
            },
            {
                "heading": "Key Concepts",
                "content": key_concepts or "Revise NCERT definitions, properties, and core examples for this topic.",
                "isKey": True
            },
            {
                "heading": "Important Reactions / Processes",
                "content": reactions or f"Focus on named reactions/processes and exceptions from NCERT {chapter_title}.",
                "isKey": False
            }
        ],
        "keyPoints": key_points,
        "formulas": [],
        "mnemonic": mnemonic_hint,
        "mnemonicSourceAvailable": bool(mnemonic_hint),
        "funFact": f"This topic frequently appears as direct NCERT-based conceptual questions in {subject}.",
        "diagrams": [],
        "examTip": f"Prioritize NCERT lines and examples from {chapter_title}; exam questions are often direct.",
        "ncertExactLines": [],
        "commonErrors": [
            "Skipping NCERT exceptions and highlighted notes.",
            "Confusing similar terms/processes without comparing them side by side."
        ]
    }


# ── REST ENDPOINTS ────────────────────────────────────────────────────────

async def discover_chapters_from_rag(subject: str, class_num: str) -> list:
    """
    Uses RAG to discover actual chapters from NCERT textbooks.
    Queries vector store for chapter info from the actual PDF content.
    """
    search_queries = [
        f"NCERT {subject} Class {class_num} chapter list contents",
        f"{subject} Class {class_num} textbook chapters topics",
        f"NCERT {class_num} {subject} unit chapter names"
    ]

    all_text = []
    for query in search_queries:
        results = vector_store.query(query_text=query, top_k=8)
        for r in results:
            meta = r.get("metadata", {})
            source = meta.get("source", "").lower()
            text = meta.get("text", "")
            # only NCERT sources
            if text and any(kw in source for kw in
                            ["ncert", "part", "class", "11", "12", "bio", "chem",
                             "phy", "textbook", "cbse"]):
                all_text.append(text)

    context = "\n\n".join(all_text[:10])[:4000]

    system_prompt = f"""You are analyzing NCERT {subject} Class {class_num} textbook content.
From the provided textbook content, extract the actual chapter list.
Return ONLY valid JSON array — no backslashes in values:
[
  {{
    "chapter": 1,
    "title": "Exact chapter title from textbook",
    "weightage": "very_high|high|medium|low",
    "examMarks": "4-8 marks",
    "tags": ["key topic 1", "key topic 2", "key topic 3"],
    "pyqCount": 45,
    "difficulty": "hard|medium|easy"
  }}
]

Base weightage on actual JEE/NEET exam importance.
Return chapters in order as they appear in the textbook.
If content is limited, generate the standard NCERT {subject} Class {class_num} chapters."""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Textbook content:\n{context}\n\nGenerate chapter list for NCERT {subject} Class {class_num}")
        ])

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            chapters = json.loads(match.group())
            return chapters
    except Exception as e:
        print(f"[NCERT] Chapter discovery error: {e}")

    return []


async def discover_subtopics_from_rag(
    subject: str, class_num: str, chapter_title: str
) -> list:
    """
    Discovers actual subtopics within a chapter from NCERT content.
    Uses RAG to find what topics are actually covered.
    """
    search_queries = [
        f"NCERT {subject} {chapter_title} topics subtopics sections",
        f"{subject} {chapter_title} headings subheadings ncert",
        f"{chapter_title} important concepts class {class_num} ncert"
    ]

    chunks = []
    for query in search_queries:
        results = vector_store.query(query_text=query, top_k=10)
        for result in results:
            metadata = result.get("metadata") or {}
            text = metadata.get("text", "")
            source = metadata.get("source", "").lower()
            if text and ("ncert" in source or "class" in source or subject.lower() in source):
                chunks.append(text)

    context = "\n\n".join(chunks[:20])[:8000]

    system_prompt = f"""Analyze NCERT {subject} Class {class_num} - Chapter: {chapter_title}
Extract the exact subtopics/sections covered in this chapter.
Return ONLY valid JSON array:
[
  {{
    "subtopic": "Exact subtopic name from NCERT",
    "isKeyTopic": true,
    "estimatedReadTime": "5-8 min",
    "examImportance": "high|medium|low"
  }}
]
Include all important subtopics that appear in NCERT. Order by textbook flow and return at least 6 subtopics whenever the chapter has enough material."""

    try:
        response = main_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Chapter content:\n{context}")
        ])

        text = response.content.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[NCERT] Subtopic discovery error: {e}")

    return []

@ncertRouter.get("/ncert/catalog")
async def getCatalog(examTarget: str = "JEE"):
    """Backward-compatible catalog endpoint using cached RAG discovery."""
    db = get_db()
    subjects = ["Chemistry"] if examTarget == "JEE" else ["Chemistry", "Biology"]
    payload = {}
    for subject in subjects:
        subject_payload = {}
        for class_num in ["11", "12"]:
            cached = await db.ncert_chapters.find_one(
                {"cacheKey": f"chapters_{subject}_{class_num}"},
                {"_id": 0, "chapters": 1}
            )
            subject_payload[f"Class {class_num}"] = (cached or {}).get("chapters", [])
        payload[subject] = subject_payload
    return {"message": "catalog", "payload": payload}


@ncertRouter.get("/ncert/chapter/{subject}/{classNum}/{chapterNum}")
async def getChapterTopics(subject: str, classNum: str, chapterNum: int):
    """Returns chapter metadata and topic list from discovered cache."""
    db = get_db()
    cached = await db.ncert_chapters.find_one(
        {"cacheKey": f"chapters_{subject}_{classNum}"},
        {"_id": 0, "chapters": 1}
    )
    chapters = (cached or {}).get("chapters", [])
    chapter = next((c for c in chapters if c.get("chapter") == chapterNum), None)

    if not chapter:
        return {"message": "not found", "payload": None}

    return {"message": "chapter", "payload": chapter}


class GenerateRequest(BaseModel):
    subject: str
    classNum: str
    chapterNum: int
    topic: str
    examTarget: str = "JEE"


@ncertRouter.post("/ncert/generate")
async def generateContent(req: GenerateRequest):
    """
    Generates and caches content for a topic.
    GLOBAL CACHE — once generated by any user, stored for ALL users forever.
    """
    db = get_db()

    class_key = f"Class {req.classNum}"
    chapters_cache = await db.ncert_chapters.find_one(
        {"cacheKey": f"chapters_{req.subject}_{req.classNum}"},
        {"_id": 0, "chapters": 1}
    )
    chapters = (chapters_cache or {}).get("chapters", [])
    chapter_info = next((c for c in chapters if c.get("chapter") == req.chapterNum), None)
    if not chapter_info:
        chapter_info = {"chapter": req.chapterNum, "title": req.topic}

    # cache key — global for all users
    cache_key = f"{req.subject}_{req.classNum}_{req.chapterNum}_{req.topic}".replace(" ", "_").lower()

    # check global cache
    cached = await db.ncert_content.find_one(
        {"cacheKey": cache_key},
        {"_id": 0}
    )

    if cached and not is_placeholder_content(cached):
        if "mnemonicSourceAvailable" not in cached:
            refreshed = dict(cached)
            refreshed["mnemonic"] = ""
            refreshed["mnemonicSourceAvailable"] = False
            await db.ncert_content.update_one(
                {"cacheKey": cache_key},
                {"$set": {
                    "mnemonic": "",
                    "mnemonicSourceAvailable": False
                }}
            )
            cached = refreshed
        print(f"[NCERT] Cache HIT: {cache_key}")
        return {"message": "content", "payload": cached, "cached": True}

    # generate new content
    print(f"[NCERT] Generating: {req.subject} {req.classNum} Ch{req.chapterNum} - {req.topic}")
    content = await generate_topic_content(
        req.subject, class_key, chapter_info, req.topic, db
    )

    # store globally (upsert avoids duplicates from parallel requests)
    content["cacheKey"] = cache_key
    content["generatedAt"] = now()
    content["generatedCount"] = int((cached or {}).get("generatedCount", 0)) + 1

    await db.ncert_content.update_one(
        {"cacheKey": cache_key},
        {"$set": content},
        upsert=True
    )
    content.pop("_id", None)

    print(f"[NCERT] Generated and cached: {cache_key}")
    return {"message": "content", "payload": content, "cached": False}


@ncertRouter.get("/ncert/content/{cacheKey}")
async def getCachedContent(cacheKey: str):
    """Get pre-generated content by cache key."""
    db = get_db()
    content = await db.ncert_content.find_one({"cacheKey": cacheKey}, {"_id": 0})
    if not content:
        return {"message": "not cached", "payload": None}
    if "mnemonicSourceAvailable" not in content:
        content["mnemonic"] = ""
        content["mnemonicSourceAvailable"] = False
        await db.ncert_content.update_one(
            {"cacheKey": cacheKey},
            {"$set": {
                "mnemonic": "",
                "mnemonicSourceAvailable": False
            }}
        )
    return {"message": "content", "payload": content}


# ── USER PROGRESS ─────────────────────────────────────────────────────────

class SaveProgressRequest(BaseModel):
    userId: str
    subject: str
    classNum: str
    chapterNum: int
    chapterTitle: str
    completedTopics: List[str]
    totalTopics: int
    startDate: Optional[str] = None
    targetDate: Optional[str] = None
    revisionCount: int = 0


@ncertRouter.post("/ncert/progress")
async def saveProgress(req: SaveProgressRequest):
    """Save user's reading progress."""
    db = get_db()

    progress_key = f"{req.userId}_{req.subject}_{req.classNum}_{req.chapterNum}"
    progress_pct = int((len(req.completedTopics) / max(req.totalTopics, 1)) * 100)

    await db.ncert_progress.update_one(
        {"progressKey": progress_key},
        {"$set": {
            "userId": req.userId,
            "subject": req.subject,
            "classNum": req.classNum,
            "chapterNum": req.chapterNum,
            "chapterTitle": req.chapterTitle,
            "completedTopics": req.completedTopics,
            "totalTopics": req.totalTopics,
            "progressPercent": progress_pct,
            "startDate": req.startDate or now(),
            "targetDate": req.targetDate,
            "revisionCount": req.revisionCount,
            "lastRead": now(),
            "isCompleted": progress_pct == 100
        }},
        upsert=True
    )

    return {"message": "progress saved", "payload": {"progressPercent": progress_pct}}


@ncertRouter.get("/ncert/progress/{userId}")
async def getUserProgress(userId: str):
    """Get all chapter progress for a user."""
    db = get_db()
    progress = await db.ncert_progress.find(
        {"userId": userId}, {"_id": 0}
    ).to_list(200)
    return {"message": "progress", "payload": progress}


@ncertRouter.post("/ncert/progress/{userId}/{subject}/{classNum}/{chapterNum}/reset")
async def resetProgress(userId: str, subject: str, classNum: str, chapterNum: int):
    """Reset chapter progress for revision."""
    db = get_db()
    progress_key = f"{userId}_{subject}_{classNum}_{chapterNum}"
    await db.ncert_progress.update_one(
        {"progressKey": progress_key},
        {"$set": {
            "completedTopics": [],
            "progressPercent": 0,
            "isCompleted": False
        },
        "$inc": {"revisionCount": 1}}
    )
    return {"message": "progress reset"}


# ── HIGHLIGHTS ────────────────────────────────────────────────────────────

class SaveHighlightRequest(BaseModel):
    userId: str
    subject: str
    classNum: str
    chapterNum: int
    chapterTitle: str
    topic: str
    highlightedText: str
    color: str = "yellow"    # yellow | green | blue | pink


@ncertRouter.post("/ncert/highlight")
async def saveHighlight(req: SaveHighlightRequest):
    """Save a highlighted text passage."""
    db = get_db()
    highlight = {
        "highlightId": str(uuid.uuid4()),
        "userId": req.userId,
        "subject": req.subject,
        "classNum": req.classNum,
        "chapterNum": req.chapterNum,
        "chapterTitle": req.chapterTitle,
        "topic": req.topic,
        "highlightedText": req.highlightedText,
        "color": req.color,
        "savedAt": now()
    }
    await db.ncert_highlights.insert_one(highlight)
    highlight.pop("_id", None)
    return {"message": "highlight saved", "payload": highlight}


@ncertRouter.get("/ncert/highlights/{userId}")
async def getHighlights(userId: str, subject: Optional[str] = None):
    """Get all highlights for a user."""
    db = get_db()
    query = {"userId": userId}
    if subject:
        query["subject"] = subject

    highlights = await db.ncert_highlights.find(
        query, {"_id": 0}
    ).sort("savedAt", -1).to_list(500)

    return {"message": "highlights", "payload": highlights}


@ncertRouter.delete("/ncert/highlight/{highlightId}")
async def deleteHighlight(highlightId: str, userId: str):
    """Delete a highlight."""
    db = get_db()
    await db.ncert_highlights.delete_one(
        {"highlightId": highlightId, "userId": userId}
    )
    return {"message": "deleted"}
