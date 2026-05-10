import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("repo_rag_api")

app = FastAPI(
    title="Repo RAG API",
    description="Analyze Java/Spring Boot repositories using RAG.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

MAX_FILE_SIZE_BYTES = 200_000
MAX_FILES = 120
RETRIEVAL_K = 8

IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
    ".idea",
}

CODE_EXTENSIONS = {
    ".java",
    ".yml",
    ".yaml",
    ".properties",
    ".xml",
    ".sql",
    ".md",
}

IMPORTANT_FILE_PATTERNS = {
    "readme": 10,

    "controller": 10,
    "restcontroller": 10,

    "service": 9,

    "repository": 8,
    "jparepository": 8,

    "entity": 7,

    "config": 7,
    "application": 7,

    "pom.xml": 10,
    "application.yml": 9,
    "application.properties": 9,

    "dto": 4,
    "mapper": 4,

    "test": 1,
}

IGNORED_METHODS = {
    "hashcode",
    "equals",
    "tostring",
    "getter",
    "setter",
}

INDEXED_REPOS: dict[str, dict[str, Any]] = {}

EMBEDDINGS: HuggingFaceEmbeddings | None = None


class RepoRequest(BaseModel):
    repo_url: str = Field(
        ...,
        description="Public Git repository URL",
    )


def validate_repo_url(repo_url: str) -> str:
    parsed = urlparse(repo_url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="repo_url must be a valid Git repository URL",
        )

    return repo_url


def clone_repo(repo_url: str, destination: Path) -> None:
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(destination)],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )

    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Repository clone timed out",
        ) from exc

    except subprocess.CalledProcessError as exc:
        message = (
            exc.stderr.strip()
            or exc.stdout.strip()
            or "Unable to clone repository"
        )

        raise HTTPException(
            status_code=400,
            detail=message,
        ) from exc


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def get_document_priority(source: str) -> int:
    source_lower = source.lower()

    for keyword, score in IMPORTANT_FILE_PATTERNS.items():
        if keyword in source_lower:
            return score

    return 5


def load_repo_documents(repo_path: Path) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(repo_path.rglob("*")):
        if len(documents) >= MAX_FILES:
            break

        relative_path = path.relative_to(repo_path)

        if not path.is_file():
            continue

        if is_ignored(relative_path):
            continue

        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue

        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue

        try:
            content = path.read_text(encoding="utf-8")

        except UnicodeDecodeError:
            continue

        if not content.strip():
            continue

        # README boost
        if relative_path.name.lower().startswith("readme"):
            content = "PROJECT OVERVIEW\n\n" + content

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(relative_path),
                    "priority": get_document_priority(str(relative_path)),
                },
            )
        )

    if not documents:
        raise HTTPException(
            status_code=400,
            detail="No readable repository files found",
        )
    print(documents[0])
    return documents


def extract_java_methods(document: Document) -> list[dict[str, str]]:
    pattern = (
        r"(?:public|private|protected)"
        r"\s+(?:static\s+)?"
        r"[\w<>\[\]]+\s+"
        r"([A-Za-z_][\w]*)"
        r"\s*\(([^)]*)\)"
    )

    methods: list[dict[str, str]] = []

    for match in re.finditer(pattern, document.page_content):
        name = match.group(1)

        if name.lower() in IGNORED_METHODS:
            continue

        # Ignore constructors
        if name[0].isupper():
            continue

        args = " ".join(match.group(2).split())

        methods.append(
            {
                "file": document.metadata.get("source", "unknown"),
                "name": name,
                "signature": f"{name}({args})",
                "description": "Detected from Java source signature.",
            }
        )

    return methods


def extract_repo_insights(documents: list[Document]) -> dict[str, Any]:
    file_types = Counter(
        Path(doc.metadata.get("source", "")).suffix
        or "[no extension]"
        for doc in documents
    )

    total_lines = sum(
        doc.page_content.count("\n") + 1
        for doc in documents
    )

    methods: list[dict[str, str]] = []

    complexity_markers = Counter()

    for document in documents:
        source = document.metadata.get("source", "")

        suffix = Path(source).suffix.lower()

        if suffix == ".java":
            methods.extend(extract_java_methods(document))

        for marker in (
            "if ",
            "for ",
            "while ",
            "switch",
            "case ",
            "catch ",
        ):
            complexity_markers[marker.strip()] += (
                document.page_content.count(marker)
            )

    complexity_score = sum(complexity_markers.values())

    if complexity_score < 25:
        complexity_level = "low"

    elif complexity_score < 100:
        complexity_level = "moderate"

    else:
        complexity_level = "high"

    ranked_methods = sorted(
        methods,
        key=lambda method: (
            "controller" in method["file"].lower(),
            "service" in method["file"].lower(),
            "repository" in method["file"].lower(),
        ),
        reverse=True,
    )

    return {
        "file_count": len(documents),
        "total_lines": total_lines,
        "file_types": dict(file_types.most_common()),
        "methods": ranked_methods[:40],
        "complexity": {
            "level": complexity_level,
            "score": complexity_score,
            "signals": dict(complexity_markers.most_common()),
        },
    }


def get_embeddings() -> HuggingFaceEmbeddings:
    global EMBEDDINGS

    if EMBEDDINGS is None:
        EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return EMBEDDINGS


def build_vector_store(documents: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return FAISS.from_documents(
        chunks,
        get_embeddings(),
    )


def update_repo_index(repo_url: str) -> dict[str, Any]:
    temp_dir = Path(tempfile.mkdtemp(prefix="repo-rag-"))

    repo_dir = temp_dir / "repo"

    try:
        clone_repo(repo_url, repo_dir)

        documents = load_repo_documents(repo_dir)

        insights = extract_repo_insights(documents)

        vector_store = build_vector_store(documents)

        INDEXED_REPOS[repo_url] = {
            "vector_store": vector_store,
            "document_count": len(documents),
            "insights": insights,
        }

        return {
            "repo_url": repo_url,
            "status": "indexed",
            "document_count": len(documents),
            "indexed_methods": len(insights["methods"]),
            "complexity": insights["complexity"],
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def parse_json_response(answer: str) -> dict[str, Any]:
    try:
        return json.loads(answer)

    except json.JSONDecodeError:
        return {
            "overview": answer,
            "architecture": "",
            "key_methods": [],
            "complexity": {},
            "noteworthy_aspects": [],
        }


def build_analysis(
    repo_url: str,
    vector_store: FAISS,
    insights: dict[str, Any],
) -> dict[str, Any]:

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not found",
        )

    question = (
        "Analyze this Java repository like a senior software architect. "
        "Explain the business purpose, Spring Boot architecture, "
        "major modules, APIs, services, repositories, entities, "
        "key methods, complexity, and noteworthy implementation details."
    )

    retrieved_docs = vector_store.similarity_search(
        question,
        k=30,
    )

    docs = sorted(
        retrieved_docs,
        key=lambda doc: doc.metadata.get("priority", 5),
        reverse=True,
    )[:RETRIEVAL_K]

    is_spring_boot = any(
        "@SpringBootApplication" in doc.page_content
        or "@RestController" in doc.page_content
        or "JpaRepository" in doc.page_content
        for doc in docs
    )

    framework_hint = (
        "This appears to be a Spring Boot Java application."
        if is_spring_boot
        else "Framework is unclear."
    )

    context = "\n\n".join(
        f"File: {doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in docs
    )

    static_insights = json.dumps(insights, indent=2)

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior Java software architect. "
            "Use only the supplied repository context. "
            "Focus on Spring Boot architecture, APIs, "
            "services, repositories, entities, "
            "business functionality, and implementation details. "
            "Return valid JSON only.",
        ),
        (
            "human",
            "Repository URL: {repo_url}\n\n"
            "Framework hint:\n{framework_hint}\n\n"
            "Static insights:\n{static_insights}\n\n"
            "Retrieved context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "If the repository contains controllers, services, "
            "repositories, entities, or Spring annotations, "
            "explain the layered architecture clearly.\n\n"
            "Return this JSON shape:\n"
            "{{\n"
            '  "overview": "project purpose",\n'
            '  "architecture": "high level architecture",\n'
            '  "key_methods": [\n'
            "    {{\n"
            '      "file": "path",\n'
            '      "signature": "method(args)",\n'
            '      "description": "what it does"\n'
            "    }}\n"
            "  ],\n"
            '  "complexity": {{\n'
            '    "level": "low|moderate|high",\n'
            '    "explanation": "why"\n'
            "  }},\n"
            '  "noteworthy_aspects": [\n'
            '    "important implementation details"\n'
            "  ]\n"
            "}}",
        ),
    ]
)

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "repo_url": repo_url,
            "framework_hint": framework_hint,
            "static_insights": static_insights,
            "context": context,
            "question": question,
        }
    )

    structured_analysis = parse_json_response(answer)

    return {
        "repo_url": repo_url,
        "analysis": structured_analysis,
        "static_insights": insights,
        "retrieved_files": sorted(
            {
                doc.metadata.get("source", "unknown")
                for doc in docs
            }
        ),
    }


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Repo RAG API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embeddings/update")
def update_embeddings(request: RepoRequest) -> dict[str, Any]:
    repo_url = validate_repo_url(request.repo_url)

    return update_repo_index(repo_url)


@app.get("/analysis")
def analyze_repo(
    repo_url: str = Query(
        ...,
        description="Public Git repository URL",
    ),
) -> dict[str, Any]:

    repo_url = validate_repo_url(repo_url)

    indexed_repo = INDEXED_REPOS.get(repo_url)

    if not indexed_repo:
        raise HTTPException(
            status_code=404,
            detail="Repository is not indexed yet. Call /embeddings/update first.",
        )

    return build_analysis(
        repo_url,
        indexed_repo["vector_store"],
        indexed_repo["insights"],
    )