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
from datetime import datetime
import json, re, uuid

ncertRouter = APIRouter()


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


# ── NCERT CHAPTER CATALOG ─────────────────────────────────────────────────
# Organized by exam, subject, class, chapter with weightage data
# Based on real NTA trend analysis

NCERT_CATALOG = {
    "Chemistry": {
        "Class 11": [
            {"chapter": 1, "title": "Some Basic Concepts of Chemistry",
             "weightage": "high", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["mole concept", "stoichiometry", "concentration"],
             "pyqCount": 45, "difficulty": "medium"},
            {"chapter": 2, "title": "Structure of Atom",
             "weightage": "high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["quantum numbers", "orbitals", "electronic config"],
             "pyqCount": 62, "difficulty": "medium"},
            {"chapter": 3, "title": "Classification of Elements",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "4-6",
             "tags": ["periodic table", "trends", "periodicity"],
             "pyqCount": 38, "difficulty": "easy"},
            {"chapter": 4, "title": "Chemical Bonding and Molecular Structure",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["VSEPR", "hybridization", "MO theory", "hydrogen bond"],
             "pyqCount": 78, "difficulty": "hard"},
            {"chapter": 5, "title": "States of Matter",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["gas laws", "kinetic theory", "van der Waals"],
             "pyqCount": 35, "difficulty": "medium"},
            {"chapter": 6, "title": "Thermodynamics",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["enthalpy", "entropy", "Gibbs energy", "Hess law"],
             "pyqCount": 85, "difficulty": "hard"},
            {"chapter": 7, "title": "Equilibrium",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["Kp", "Kc", "Le Chatelier", "buffer", "pH"],
             "pyqCount": 92, "difficulty": "hard"},
            {"chapter": 8, "title": "Redox Reactions",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["oxidation state", "balancing", "electrochemistry basics"],
             "pyqCount": 42, "difficulty": "medium"},
            {"chapter": 9, "title": "Hydrogen",
             "weightage": "low", "jeeMarks": "0-2", "neetMarks": "2-4",
             "tags": ["hydrides", "water", "heavy water", "hydrogen peroxide"],
             "pyqCount": 22, "difficulty": "easy"},
            {"chapter": 10, "title": "The s-Block Elements",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "4-6",
             "tags": ["alkali metals", "alkaline earth", "anomalous behaviour"],
             "pyqCount": 48, "difficulty": "medium"},
            {"chapter": 11, "title": "The p-Block Elements (Group 13-14)",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["boron family", "carbon family", "allotropes"],
             "pyqCount": 35, "difficulty": "medium"},
            {"chapter": 12, "title": "Organic Chemistry: Basic Principles",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["IUPAC", "isomerism", "electronic effects", "reactive intermediates"],
             "pyqCount": 95, "difficulty": "hard"},
            {"chapter": 13, "title": "Hydrocarbons",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["alkanes", "alkenes", "alkynes", "aromatic", "reactions"],
             "pyqCount": 72, "difficulty": "medium"},
            {"chapter": 14, "title": "Environmental Chemistry",
             "weightage": "low", "jeeMarks": "0-2", "neetMarks": "2-4",
             "tags": ["pollution", "ozone", "smog", "green chemistry"],
             "pyqCount": 18, "difficulty": "easy"},
        ],
        "Class 12": [
            {"chapter": 1, "title": "The Solid State",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["crystal systems", "defects", "packing efficiency"],
             "pyqCount": 40, "difficulty": "medium"},
            {"chapter": 2, "title": "Solutions",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["colligative properties", "Raoult's law", "molarity"],
             "pyqCount": 58, "difficulty": "medium"},
            {"chapter": 3, "title": "Electrochemistry",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["Nernst equation", "electrolysis", "Faraday", "EMF"],
             "pyqCount": 80, "difficulty": "hard"},
            {"chapter": 4, "title": "Chemical Kinetics",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["rate law", "Arrhenius", "order", "activation energy"],
             "pyqCount": 88, "difficulty": "hard"},
            {"chapter": 5, "title": "Surface Chemistry",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["adsorption", "colloids", "catalysis", "emulsions"],
             "pyqCount": 35, "difficulty": "medium"},
            {"chapter": 6, "title": "General Principles of Isolation of Metals",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["metallurgy", "reduction", "refining"],
             "pyqCount": 30, "difficulty": "medium"},
            {"chapter": 7, "title": "The p-Block Elements (Group 15-18)",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["nitrogen family", "oxygen family", "halogens", "noble gases"],
             "pyqCount": 90, "difficulty": "hard"},
            {"chapter": 8, "title": "The d and f-Block Elements",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["transition metals", "lanthanides", "actinides", "properties"],
             "pyqCount": 65, "difficulty": "medium"},
            {"chapter": 9, "title": "Coordination Compounds",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["IUPAC naming", "isomerism", "CFT", "VBT"],
             "pyqCount": 95, "difficulty": "hard"},
            {"chapter": 10, "title": "Haloalkanes and Haloarenes",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["SN1", "SN2", "elimination", "Grignard"],
             "pyqCount": 70, "difficulty": "hard"},
            {"chapter": 11, "title": "Alcohols, Phenols and Ethers",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["reactions", "acidity", "preparation", "Williamson"],
             "pyqCount": 65, "difficulty": "medium"},
            {"chapter": 12, "title": "Aldehydes, Ketones and Carboxylic Acids",
             "weightage": "very_high", "jeeMarks": "4-8", "neetMarks": "4-8",
             "tags": ["nucleophilic addition", "aldol", "Cannizzaro", "acidity"],
             "pyqCount": 92, "difficulty": "hard"},
            {"chapter": 13, "title": "Amines",
             "weightage": "high", "jeeMarks": "4-6", "neetMarks": "4-6",
             "tags": ["basicity", "diazonium", "coupling reactions", "Hoffmann"],
             "pyqCount": 68, "difficulty": "hard"},
            {"chapter": 14, "title": "Biomolecules",
             "weightage": "high", "jeeMarks": "2-4", "neetMarks": "4-8",
             "tags": ["carbohydrates", "proteins", "vitamins", "nucleic acids"],
             "pyqCount": 55, "difficulty": "medium"},
            {"chapter": 15, "title": "Polymers",
             "weightage": "medium", "jeeMarks": "2-4", "neetMarks": "2-4",
             "tags": ["addition", "condensation", "natural polymers", "rubber"],
             "pyqCount": 30, "difficulty": "easy"},
            {"chapter": 16, "title": "Chemistry in Everyday Life",
             "weightage": "low", "jeeMarks": "0-2", "neetMarks": "2-4",
             "tags": ["drugs", "soaps", "food preservatives", "dyes"],
             "pyqCount": 20, "difficulty": "easy"},
        ]
    },
    "Biology": {
        "Class 11": [
            {"chapter": 1, "title": "The Living World",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["taxonomy", "nomenclature", "classification"],
             "pyqCount": 25, "difficulty": "easy"},
            {"chapter": 2, "title": "Biological Classification",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["five kingdoms", "viruses", "bacteria", "fungi"],
             "pyqCount": 48, "difficulty": "medium"},
            {"chapter": 3, "title": "Plant Kingdom",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["algae", "bryophytes", "pteridophytes", "gymnosperms"],
             "pyqCount": 52, "difficulty": "medium"},
            {"chapter": 4, "title": "Animal Kingdom",
             "weightage": "high", "neetMarks": "4-8",
             "tags": ["phyla", "chordates", "classification", "features"],
             "pyqCount": 65, "difficulty": "medium"},
            {"chapter": 5, "title": "Morphology of Flowering Plants",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["root", "stem", "leaf", "flower", "fruit", "seed"],
             "pyqCount": 50, "difficulty": "medium"},
            {"chapter": 6, "title": "Anatomy of Flowering Plants",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["meristems", "tissues", "anatomy"],
             "pyqCount": 32, "difficulty": "medium"},
            {"chapter": 7, "title": "Structural Organisation in Animals",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["tissues", "cockroach anatomy", "earthworm"],
             "pyqCount": 30, "difficulty": "medium"},
            {"chapter": 8, "title": "Cell: The Unit of Life",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["cell organelles", "prokaryotic", "eukaryotic", "membrane"],
             "pyqCount": 80, "difficulty": "medium"},
            {"chapter": 9, "title": "Biomolecules",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["carbohydrates", "proteins", "lipids", "enzymes", "nucleic acids"],
             "pyqCount": 85, "difficulty": "hard"},
            {"chapter": 10, "title": "Cell Cycle and Cell Division",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["mitosis", "meiosis", "phases", "significance"],
             "pyqCount": 90, "difficulty": "hard"},
            {"chapter": 11, "title": "Transport in Plants",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["osmosis", "diffusion", "xylem", "phloem"],
             "pyqCount": 35, "difficulty": "medium"},
            {"chapter": 12, "title": "Mineral Nutrition",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["macro", "micro nutrients", "deficiency", "nitrogen fixation"],
             "pyqCount": 28, "difficulty": "easy"},
            {"chapter": 13, "title": "Photosynthesis in Higher Plants",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["light reactions", "Calvin cycle", "C4", "CAM", "photorespiration"],
             "pyqCount": 88, "difficulty": "hard"},
            {"chapter": 14, "title": "Respiration in Plants",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["glycolysis", "Krebs cycle", "ETC", "fermentation"],
             "pyqCount": 72, "difficulty": "hard"},
            {"chapter": 15, "title": "Plant Growth and Development",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["hormones", "auxin", "gibberellin", "photoperiodism"],
             "pyqCount": 40, "difficulty": "medium"},
            {"chapter": 16, "title": "Digestion and Absorption",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["alimentary canal", "enzymes", "absorption", "disorders"],
             "pyqCount": 82, "difficulty": "hard"},
            {"chapter": 17, "title": "Breathing and Exchange of Gases",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["lungs", "respiration mechanism", "transport", "disorders"],
             "pyqCount": 65, "difficulty": "medium"},
            {"chapter": 18, "title": "Body Fluids and Circulation",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["blood", "heart", "ECG", "blood pressure", "disorders"],
             "pyqCount": 88, "difficulty": "hard"},
            {"chapter": 19, "title": "Excretory Products and their Elimination",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["nephron", "urine formation", "osmoregulation", "disorders"],
             "pyqCount": 85, "difficulty": "hard"},
            {"chapter": 20, "title": "Locomotion and Movement",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["muscles", "skeletal system", "joints", "disorders"],
             "pyqCount": 42, "difficulty": "medium"},
            {"chapter": 21, "title": "Neural Control and Coordination",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["neuron", "synapse", "brain", "reflex", "action potential"],
             "pyqCount": 90, "difficulty": "hard"},
            {"chapter": 22, "title": "Chemical Coordination and Integration",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["hormones", "glands", "feedback", "disorders"],
             "pyqCount": 85, "difficulty": "hard"},
        ],
        "Class 12": [
            {"chapter": 1, "title": "Reproduction in Organisms",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["asexual", "sexual reproduction", "life cycles"],
             "pyqCount": 30, "difficulty": "easy"},
            {"chapter": 2, "title": "Sexual Reproduction in Flowering Plants",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["pollination", "fertilization", "double fertilization", "fruits"],
             "pyqCount": 85, "difficulty": "hard"},
            {"chapter": 3, "title": "Human Reproduction",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["gametogenesis", "menstrual cycle", "fertilization", "development"],
             "pyqCount": 90, "difficulty": "hard"},
            {"chapter": 4, "title": "Reproductive Health",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["contraception", "STDs", "infertility", "ART"],
             "pyqCount": 35, "difficulty": "easy"},
            {"chapter": 5, "title": "Principles of Inheritance and Variation",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["Mendel", "monohybrid", "dihybrid", "codominance", "linkage"],
             "pyqCount": 95, "difficulty": "hard"},
            {"chapter": 6, "title": "Molecular Basis of Inheritance",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["DNA replication", "transcription", "translation", "genetic code"],
             "pyqCount": 92, "difficulty": "hard"},
            {"chapter": 7, "title": "Evolution",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["Darwin", "natural selection", "speciation", "human evolution"],
             "pyqCount": 68, "difficulty": "medium"},
            {"chapter": 8, "title": "Human Health and Disease",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["immunity", "pathogens", "cancer", "drugs", "AIDS"],
             "pyqCount": 88, "difficulty": "hard"},
            {"chapter": 9, "title": "Strategies for Enhancement in Food Production",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["plant breeding", "tissue culture", "biofortification"],
             "pyqCount": 35, "difficulty": "easy"},
            {"chapter": 10, "title": "Microbes in Human Welfare",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["fermentation", "antibiotics", "biogas", "sewage"],
             "pyqCount": 55, "difficulty": "medium"},
            {"chapter": 11, "title": "Biotechnology: Principles and Processes",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["recombinant DNA", "PCR", "restriction enzymes", "cloning"],
             "pyqCount": 90, "difficulty": "hard"},
            {"chapter": 12, "title": "Biotechnology and Its Applications",
             "weightage": "very_high", "neetMarks": "4-8",
             "tags": ["GMO", "gene therapy", "insulin", "Bt crops", "biopiracy"],
             "pyqCount": 85, "difficulty": "hard"},
            {"chapter": 13, "title": "Organisms and Populations",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["ecology", "population growth", "interactions", "adaptations"],
             "pyqCount": 65, "difficulty": "medium"},
            {"chapter": 14, "title": "Ecosystem",
             "weightage": "high", "neetMarks": "4-6",
             "tags": ["food chain", "energy flow", "nutrient cycling", "succession"],
             "pyqCount": 60, "difficulty": "medium"},
            {"chapter": 15, "title": "Biodiversity and Conservation",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["hotspots", "extinction", "conservation strategies", "IUCN"],
             "pyqCount": 40, "difficulty": "easy"},
            {"chapter": 16, "title": "Environmental Issues",
             "weightage": "medium", "neetMarks": "2-4",
             "tags": ["pollution", "ozone", "global warming", "deforestation"],
             "pyqCount": 32, "difficulty": "easy"},
        ]
    }
}


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
    return "\n\n".join(all_texts[:60])


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
        query_text=f"mnemonic trick tip {topic} {subject} remember",
        top_k=4
    )
    mnemonic_context = "\n\n".join([
        r["metadata"].get("text", "") for r in mnemonic_results
        if r.get("metadata") and r["metadata"].get("text")
    ])

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
{ncert_context[:20000]}

Mnemonic/Tips Available:
{mnemonic_context[:4000]}

Cover ALL points from NCERT thoroughly. Do not skip minor exceptions, notes, or examples that can appear in exams.
If content is long, still prioritize completeness over brevity and preserve exam-critical details."""

    try:
        response = main_llm.invoke([
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
            return json.loads(json_str)
    except Exception as e:
        print(f"[NCERT] Generation error for {topic}: {e}")

    # deterministic fallback for rate limits/JSON errors:
    # use retrieved NCERT chunks so user still gets useful content.
    fallback_lines = [ln.strip() for ln in ncert_context.split("\n") if ln.strip()]
    short_context = " ".join(fallback_lines)[:2200]
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
        mnemonic_hint = mnemonic_context.strip().split("\n")[0][:180]

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

@ncertRouter.get("/ncert/catalog")
async def getCatalog(examTarget: str = "JEE"):
    """
    Returns full NCERT chapter catalog filtered by exam target.
    JEE → Chemistry only
    NEET → Chemistry + Biology
    """
    if examTarget == "JEE":
        catalog = {"Chemistry": NCERT_CATALOG["Chemistry"]}
    else:
        catalog = NCERT_CATALOG

    return {"message": "catalog", "payload": catalog}


@ncertRouter.get("/ncert/chapter/{subject}/{classNum}/{chapterNum}")
async def getChapterTopics(subject: str, classNum: str, chapterNum: int):
    """Returns chapter metadata and topic list."""
    class_key = f"Class {classNum}"
    chapters = NCERT_CATALOG.get(subject, {}).get(class_key, [])
    chapter = next((c for c in chapters if c["chapter"] == chapterNum), None)

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
    chapters = NCERT_CATALOG.get(req.subject, {}).get(class_key, [])
    chapter_info = next((c for c in chapters if c["chapter"] == req.chapterNum), None)

    if not chapter_info:
        return {"message": "chapter not found", "payload": None}

    # cache key — global for all users
    cache_key = f"{req.subject}_{req.classNum}_{req.chapterNum}_{req.topic}".replace(" ", "_").lower()

    # check global cache
    cached = await db.ncert_content.find_one(
        {"cacheKey": cache_key},
        {"_id": 0}
    )

    if cached and not is_placeholder_content(cached):
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