"""Deterministic evidence analysis.

For each signal family (AI, Python/backend, cloud/full-stack, engineering) this module
finds where a signal appears and *how convincingly* (EvidenceLevel):

  skills / summary section ........................ KEYWORD_ONLY
  project / job entry, no implementation verb ..... PROJECT_MENTION
  project / job entry with "built/implemented..." . IMPLEMENTATION
  ...and the same entry shows >=3 related signals . DEEP_IMPLEMENTATION

Terms in education, certifications or other free text are deliberately ignored.
No LLM is involved anywhere in this file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.extraction.section_extractor import (
    Block, Sections, scannable_blocks, split_sections,
)
from src.models import Evidence, EvidenceLevel

L = EvidenceLevel


@dataclass(frozen=True)
class SignalSpec:
    name: str
    pattern: re.Pattern
    weight: float


def _s(name: str, pattern: str, weight: float = 0.0) -> SignalSpec:
    return SignalSpec(name, re.compile(pattern, re.I), weight)


IMPL_VERB = re.compile(
    r"\b(?:built|build|building|implemented|implementing|implement|developed|developing|"
    r"develop|designed|designing|design|engineered|created|creating|architected|deployed|"
    r"deploying|integrated|integrating|wrote|authored|optimi[sz]ed|containeri[sz]ed|"
    r"automated|orchestrated|migrated|configured|launched|shipped|trained|fine-?tuned|"
    r"refactored|set up)\b",
    re.I,
)
_STACK_LINE = re.compile(r"^(?:tech(?:nologies)?(?: stack)?|stack|tools|built with|technologies used)\s*[:\-]", re.I)

# --- AI / agentic family -----------------------------------------------------
AI_DEPTH_SIGNALS = ("retrieval", "embeddings", "vector_search", "tool_calling", "state",
                    "orchestration", "evaluation", "business_logic")
AI_SPECS: tuple[SignalSpec, ...] = (
    _s("core", r"\bllms?\b|large language models?|\brag\b|retrieval[- ]augmented|\blangchain\b|"
       r"\blanggraph\b|llama[- ]?index|\bembeddings?\b|vector (?:search|stores?|databases?|db|index\w*)|"
       r"\bfaiss\b|\bchroma(?:db)?\b|\bpinecone\b|\bweaviate\b|\bqdrant\b|\bpgvector\b|\bmilvus\b|"
       r"tool[- ]?calling|function[- ]?calling|\bagentic\b|\b(?:ai|llm|autonomous|multi)[- ]agents?\b|"
       r"\bagent (?:workflow|loop|framework|graph)s?\b|\bopenai\b|\bgpt-?\d\w*|\bchatgpt\b|\bclaude\b|"
       r"\bgemini\b|\banthropic\b|hugging ?face|prompt engineering|\bcrewai\b|\bautogen\b|\bmcp\b|"
       r"semantic search|\bgenerative ai\b|\bgenai\b|\bbedrock\b|\bollama\b"),
    _s("retrieval", r"\brag\b|retriev\w*|semantic search|rerank\w*|hybrid search|\bbm25\b|knowledge base", 6),
    _s("embeddings", r"\bembeddings?\b|sentence[- ]transformers?|text-embedding", 4),
    _s("vector_search", r"vector (?:search|stores?|databases?|db|index\w*)|\bfaiss\b|\bchroma(?:db)?\b|"
       r"\bpinecone\b|\bweaviate\b|\bqdrant\b|\bpgvector\b|\bmilvus\b|similarity search", 3),
    _s("tool_calling", r"tool[- ]?calling|function[- ]?calling|tool[- ]use|custom tools?|\bmcp\b|"
       r"\bagent tools?\b|bind_tools", 6),
    _s("state", r"stateful|state management|conversation(?:al)? (?:state|memory|history)|"
       r"chat (?:memory|history)|checkpoint\w*|session (?:state|memory)|persistent (?:state|memory)|"
       r"memory (?:store|buffer|management)|state graph|state machine", 5),
    _s("orchestration", r"orchestrat\w*|\blanggraph\b|multi[- ]agent|agent(?:ic)? (?:workflow|loop|graph|pipeline)s?|"
       r"(?:agent(?:ic)?|llm|ai|multi[- ]step|stateful) workflows?|\bcrewai\b|\bautogen\b|\bdag\b|"
       r"supervisor agent|\bplanner\b", 4),
    _s("evaluation", r"evaluat\w*|\bragas\b|benchmark\w*|\bevals?\b|llm-as-(?:a-)?judge|hallucination\w*|"
       r"\bprecision\b|\brecall\b|\bf1\b|golden (?:set|dataset)|test set|\bmrr\b|\bndcg\b|\brouge\b|\bbleu\b|hit rate", 5),
    _s("business_logic", r"chunk\w*|ingest\w*|\bpars(?:e|ing|er)\b|structured output|json schema|pydantic|"
       r"guardrails?|validat\w*|\bschema\b|postgres\w*|\bsql\b|database|\bqueue\w*|\bcelery\b|scor(?:e|ing)\b|"
       r"\brank\w*|classif\w*|extraction|\bocr\b|authenticat\w*|role-based|compliance|invoice\w*|"
       r"knowledge graph|api endpoints?", 3),
    # Marks "just calling a hosted model API". Weight 0: used only for thin-wrapper detection.
    _s("api_only", r"\bopenai\b|\bgpt-?\d\w*|\bchatgpt\b|\bclaude\b|\bgemini\b|\banthropic\b|llm api|"
       r"\bprompt\w*|\bcompletions?\b|\bchat ?bots?\b|\bgenerative ai\b"),
)

# --- Python & backend (30) -----------------------------------------------------
BACKEND_SPECS: tuple[SignalSpec, ...] = (
    _s("python", r"\bpython3?\b|\bfastapi\b|\bdjango\b|\bflask\b|\bpydantic\b|\bpytest\b|\bsqlalchemy\b|"
       r"\bpandas\b|\bnumpy\b|\bpytorch\b|\bcelery\b|\basyncio\b|\buvicorn\b", 6),
    _s("framework", r"\bfastapi\b|\bdjango\b|\bflask\b|\bstarlette\b|\btornado\b", 4),
    _s("async", r"\basync(?:io)?\b|\basynchronous\b|\bawait\b|\baiohttp\b|\bhttpx\b", 3),
    _s("database", r"\bpostgres(?:ql)?\b|\bmysql\b|\bsqlite\b|\bsql\b|\bsqlalchemy\b|\balembic\b|\bmongo(?:db)?\b", 3),
    _s("redis", r"\bredis\b", 3),
    _s("rest_api", r"\brest(?:ful)?\b|api endpoints?|\bopenapi\b|\bswagger\b|\bgraphql\b|\bgrpc\b|\bapis\b", 3),
    _s("architecture", r"architecture|microservices?|design patterns?|\bmodular\b|domain-driven|service layer", 2),
    _s("testing", r"\bpytest\b|unit tests?|integration tests?|test coverage|\bunittest\b|test suite|\btdd\b|"
       r"automated tests?|\btesting\b", 3),
    _s("error_handling", r"error[- ]handling|exception[- ]handling|\bretr(?:y|ies)\b|graceful\w*|\bfallbacks?\b|"
       r"fault[- ]toleran\w*|\blogging\b", 2),
    _s("concurrency", r"concurren\w*|multi-?thread\w*|multi-?process\w*|thread pool|\bparallel\w*|\bcelery\b|"
       r"\bworkers?\b|\basyncio\b", 1),
)

# --- Cloud / deployment / full stack (15) ------------------------------------------
CLOUD_SPECS: tuple[SignalSpec, ...] = (
    _s("docker", r"\bdocker\w*|\bcontainer(?:s|ized|ised)?\b", 3),
    _s("cloud", r"\baws\b|amazon web services|\bgcp\b|google cloud|\bazure\b|\bec2\b|\bs3\b|\blambda\b|"
       r"cloud run|\bgke\b|\becs\b|\bcloud\b|vertex ai|\bbedrock\b", 3),
    _s("kubernetes", r"kubernetes|\bk8s\b|\bhelm\b", 1),
    _s("iac", r"terraform|infrastructure as code|\bpulumi\b|cloudformation", 1),
    _s("deployment", r"deploy\w*|\bci/?cd\b|github actions|\bjenkins\b|\bvercel\b|\brender\b|\bheroku\b|"
       r"\brailway\b|\bnetlify\b|in production|\bhosted\b", 2),
    _s("frontend", r"\breact(?:\.?js)?\b|next\.?js|\bvue\b|\bangular\b|\bfrontend\b|front-end|\btailwind\w*|\bsvelte\b", 3),
    _s("end_to_end", r"end[- ]to[- ]end|full[- ]?stack|\bweb app\w*|web application", 2),
)

# --- Engineering depth (5) ------------------------------------------------------------
ENGINEERING_SPECS: tuple[SignalSpec, ...] = (
    _s("testing", r"\bpytest\b|unit tests?|integration tests?|test coverage|\bunittest\b|test suite|\btdd\b|"
       r"automated tests?|\bmock\w*|\btesting\b", 1.0),
    _s("caching", r"\bcach(?:e|es|ing|ed)\b|\bmemoiz\w*|\blru\b", 0.75),
    _s("queues", r"\bqueues?\b|\bcelery\b|\brabbitmq\b|\bkafka\b|\bsqs\b|\bpub/?sub\b|message broker|"
       r"background (?:jobs?|tasks?|workers?)", 0.75),
    _s("concurrency", r"concurren\w*|multi-?thread\w*|multi-?process\w*|thread pool|\bparallel\w*|\basyncio\b", 0.5),
    _s("observability", r"observability|\bmonitoring\b|\blogging\b|\bprometheus\b|\bgrafana\b|opentelemetry|"
       r"\btracing\b|\bsentry\b|\bdatadog\b|structured logs?", 0.5),
    _s("retry_failure", r"\bretr(?:y|ies)\b|\bbackoff\b|failure handling|fault[- ]toleran\w*|graceful\w*|"
       r"circuit breaker|\bfallbacks?\b|idempoten\w*|error handling", 0.5),
    _s("architecture", r"architecture|microservices?|design patterns?|domain-driven|event-driven", 0.25),
    _s("performance", r"performance|optimi[sz]\w*|latency|throughput|\bprofil(?:ing|ed)\b|speed(?:ed)? up", 0.25),
    _s("db_indexing", r"\b(?:db|database|sql|postgres\w*)[ -]index\w*|composite index|query optimi[sz]\w*|"
       r"explain analyze|\bindexes\b", 0.25),
    _s("rate_limiting", r"rate[- ]limit\w*|\bthrottl\w*|\bquota\b", 0.25),
)

FAMILIES: dict[str, tuple[SignalSpec, ...]] = {
    "ai": AI_SPECS, "backend": BACKEND_SPECS, "cloud": CLOUD_SPECS, "engineering": ENGINEERING_SPECS,
}
_CATEGORY = {"ai": "ai_project_depth", "backend": "python_backend", "cloud": "cloud_fullstack",
             "engineering": "engineering_depth"}
_CONF = {L.KEYWORD_ONLY: 0.45, L.PROJECT_MENTION: 0.65, L.IMPLEMENTATION: 0.85, L.DEEP_IMPLEMENTATION: 0.95}


@dataclass
class SignalHit:
    signal: str
    level: EvidenceLevel
    section: str
    text: str


@dataclass
class BlockAnalysis:
    """One project/job entry as seen by a single signal family."""

    title: str
    section: str
    text: str
    levels: dict[str, EvidenceLevel] = field(default_factory=dict)  # signal -> level in this block
    deep: bool = False

    @property
    def signals(self) -> set[str]:
        return set(self.levels)

    def depth_signals(self) -> set[str]:
        return self.signals & set(AI_DEPTH_SIGNALS)

    @property
    def is_ai(self) -> bool:
        return "core" in self.levels

    @property
    def is_thin(self) -> bool:
        """AI usage that is only 'input -> hosted model API -> output'."""
        return self.is_ai and not self.depth_signals() and "api_only" in self.levels


@dataclass
class FamilyAnalysis:
    name: str
    hits: dict[str, SignalHit] = field(default_factory=dict)  # best hit per signal
    blocks: list[BlockAnalysis] = field(default_factory=list)

    def level(self, signal: str) -> EvidenceLevel:
        hit = self.hits.get(signal)
        return hit.level if hit else L.NONE

    @property
    def ai_blocks(self) -> list[BlockAnalysis]:
        return [b for b in self.blocks if b.is_ai]


@dataclass
class EvidenceAnalysis:
    sections: Sections
    ai: FamilyAnalysis
    backend: FamilyAnalysis
    cloud: FamilyAnalysis
    engineering: FamilyAnalysis
    unstructured: bool = False

    def family(self, key: str) -> FamilyAnalysis:
        return getattr(self, key)


def analyze_text(text: str) -> EvidenceAnalysis:
    """Convenience entry point: raw resume text -> evidence analysis."""
    return analyze(split_sections(text), text)


def analyze(sections: Sections, full_text: str) -> EvidenceAnalysis:
    blocks = scannable_blocks(sections, full_text)
    return EvidenceAnalysis(
        sections=sections,
        ai=_analyze_family("ai", AI_SPECS, sections, blocks),
        backend=_analyze_family("backend", BACKEND_SPECS, sections, blocks),
        cloud=_analyze_family("cloud", CLOUD_SPECS, sections, blocks),
        engineering=_analyze_family("engineering", ENGINEERING_SPECS, sections, blocks),
        unstructured=not sections.structured,
    )


def _analyze_family(name: str, specs: tuple[SignalSpec, ...], sections: Sections,
                    blocks: list[Block]) -> FamilyAnalysis:
    result = FamilyAnalysis(name)
    weights = {s.name: s.weight for s in specs}

    def record(hit: SignalHit) -> None:
        current = result.hits.get(hit.signal)
        if current is None or hit.level > current.level:
            result.hits[hit.signal] = hit

    # Skills / summary: keyword-level evidence only.
    for section in ("skills", "summary"):
        for line in sections.get(section).split("\n"):
            for spec in specs:
                if spec.pattern.search(line):
                    record(SignalHit(spec.name, L.KEYWORD_ONLY, section, line.strip()))

    # Project / experience entries: mention vs implementation vs deep implementation.
    for block in blocks:
        present: dict[str, list[int]] = {}
        for idx, item in enumerate(block.items):
            for spec in specs:
                if spec.pattern.search(item):
                    present.setdefault(spec.name, []).append(idx)
        if not present:
            continue
        verb_items = [i for i, it in enumerate(block.items) if IMPL_VERB.search(it)]
        related = {n for n in present if weights[n] > 0}
        deep = bool(verb_items) and len(related) >= 3
        analysis = BlockAnalysis(block.title, block.section, block.text, deep=deep)
        for sig, idxs in present.items():
            best_level, best_text = L.NONE, ""
            for idx in idxs:
                item = block.items[idx]
                supported = idx in verb_items or bool(verb_items) and (
                    (idx == 0 and block.has_title) or _STACK_LINE.match(item)
                )
                level = L.IMPLEMENTATION if supported else L.PROJECT_MENTION
                if supported and deep and weights[sig] > 0:
                    level = L.DEEP_IMPLEMENTATION
                if level > best_level:
                    best_level, best_text = level, item
                    if supported and idx not in verb_items:
                        # tech named in a title/stack line: pair it with the action line
                        best_text = f"{item} -> {block.items[verb_items[0]]}"
            analysis.levels[sig] = best_level
            record(SignalHit(sig, best_level, block.section, best_text))
        result.blocks.append(analysis)
    return result


# ---------------------------------------------------------------------------
# Evidence objects for the report
# ---------------------------------------------------------------------------

def hit_to_evidence(hit: SignalHit, category: str, unstructured: bool = False) -> Evidence:
    confidence = _CONF.get(hit.level, 0.3) * (0.9 if unstructured or hit.section == "unstructured" else 1.0)
    text = re.sub(r"\s+", " ", hit.text).strip()
    return Evidence(skill=hit.signal, section=hit.section, evidence_text=text[:240],
                    confidence=round(confidence, 2), level=hit.level, category=category)


def evidence_for_family(analysis: EvidenceAnalysis, key: str) -> list[Evidence]:
    """One evidence entry per signal (its best occurrence), strongest first."""
    fam = analysis.family(key)
    category = _CATEGORY[key]
    hits = [h for h in fam.hits.values() if not (key == "ai" and h.signal == "api_only")]
    hits.sort(key=lambda h: (-h.level, h.signal))
    return [hit_to_evidence(h, category, analysis.unstructured) for h in hits]


def best_ai_block(analysis: EvidenceAnalysis) -> Optional[BlockAnalysis]:
    blocks = analysis.ai.ai_blocks
    if not blocks:
        return None
    return max(blocks, key=lambda b: (b.levels["core"], len(b.depth_signals()), b.title))
