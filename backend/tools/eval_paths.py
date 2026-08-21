"""Generate learning paths for a topic set and write them as readable transcripts.

This is the measurement instrument for tuning the ranking algorithm. The thing being tuned
— does this look like a learning path to a person trying to learn the topic? — has no
automatic metric, so the output is written for a human (or a critique agent role-playing
one) to read and score. It deliberately shows the *sequence* and *stage* of each paper and
nothing about citation counts, so a reviewer cannot mistake popularity for pedagogy.

    ./.venv/bin/python tools/eval_paths.py --strategy syllabus --run r1
    ./.venv/bin/python tools/eval_paths.py --strategy anchor --model glm-5.1:cloud
    ./.venv/bin/python tools/eval_paths.py --strategy structural --topics "dropout"

Every run writes `<out>/<run>/<strategy>__<model>.md` plus a `.json` of the same data.
Provider and LLM responses are cached, so re-running a combination is cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperthread.config import load_config  # noqa: E402
from paperthread.domain.path import LearningPath  # noqa: E402
from paperthread.llm.registry import LLMClient  # noqa: E402
from paperthread.retrieval.curriculum import BuildContext, build_strategy  # noqa: E402
from paperthread.retrieval.path import LearningPathService  # noqa: E402

# Ten topics spanning the ways a learning path can go wrong: a topic whose name collides
# with another field ("transformers", "dropout"), one whose foundations predate its
# vocabulary ("diffusion models"), one that is a technique rather than a thing
# ("batch normalization"), one that is mostly recent ("RLHF"), and one whose canonical
# paper is old and rarely cited directly ("backpropagation").
DEFAULT_TOPICS = [
    "dropout in neural networks",
    "batch normalization",
    "transformers",
    "diffusion models",
    "generative adversarial networks",
    "word embeddings",
    "convolutional neural networks",
    "reinforcement learning from human feedback",
    "regularization in machine learning",
    "contrastive learning",
]


@dataclass
class TopicRun:
    topic: str
    seconds: float
    step_count: int
    degraded: bool
    confidence: float = 0.0
    confidence_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    error: str | None = None


def to_record(path: LearningPath) -> list[dict]:
    return [
        {
            "position": step.order + 1,
            "stage": step.subtopic_id or "-",
            "year": step.paper.year,
            "title": step.paper.title,
            "authors": step.paper.authors[:3],
            "teaches": step.explanation.what_it_teaches,
            "why_here": step.explanation.why_it_matters,
        }
        for step in sorted(path.steps, key=lambda s: s.order)
    ]


async def run_topic(topic: str, strategy_name: str, config, budget: int) -> TopicRun:
    started = time.monotonic()
    try:
        if strategy_name == "structural":
            # The pre-LLM pipeline, kept as the baseline every strategy is measured against.
            path = await LearningPathService(config).build(topic, limit=budget)
        else:
            context = BuildContext.create(config, LLMClient(config))
            path = await build_strategy(strategy_name, context).build(topic, budget)
    except Exception as exc:  # noqa: BLE001 - one bad topic must not end the sweep
        return TopicRun(
            topic=topic,
            seconds=time.monotonic() - started,
            step_count=0,
            degraded=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    return TopicRun(
        topic=topic,
        seconds=time.monotonic() - started,
        step_count=len(path.steps),
        degraded=path.degraded,
        confidence=round(path.confidence, 2),
        confidence_reasons=path.confidence_reasons,
        notes=path.notes,
        stages_run=path.stages_run,
        steps=to_record(path),
    )


def render(strategy: str, model: str, runs: list[TopicRun]) -> str:
    lines = [
        f"# Learning paths — strategy `{strategy}`, planning model `{model}`",
        "",
        "Each block is what a user sees after typing the topic. Read it as a learner:",
        "**would following these papers in this order teach you the topic?**",
        "",
        "`stage` is the path's own claim about each step — `prerequisite` (read before the "
        "main idea), `anchor` (the paper that IS the topic), `followup` (what came after).",
        "",
        "---",
        "",
    ]
    for run in runs:
        lines.append(f"## {run.topic}")
        lines.append("")
        if run.error:
            lines.append(f"**FAILED** — {run.error}")
            lines.append("")
            continue
        band = (
            "HIGH" if run.confidence >= 0.8
            else "MEDIUM" if run.confidence >= 0.55
            else "LOW"
        )
        lines.append(f"*{run.step_count} steps, {run.seconds:.0f}s — self-rated confidence "
                     f"**{band}** ({run.confidence:.2f})*")
        if run.confidence_reasons:
            lines.append("")
            lines.append(f"> {'; '.join(run.confidence_reasons)}")
        lines.append("")
        if not run.steps:
            lines.append("**Empty path.**")
        for step in run.steps:
            authors = ", ".join(step["authors"]) or "unknown"
            year = step["year"] or "n.d."
            lines.append(
                f"{step['position']}. **[{step['stage']}]** ({year}) *{step['title']}* — {authors}"
            )
            lines.append(f"   - teaches: {step['teaches']}")
            lines.append(f"   - why here: {step['why_here']}")
        lines.append("")
        if run.notes:
            lines.append("<details><summary>pipeline notes</summary>")
            lines.append("")
            for note in run.notes:
                lines.append(f"- {note}")
            lines.append("")
            lines.append("</details>")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="syllabus")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the primary planning model (fallbacks are kept).",
    )
    parser.add_argument("--topics", nargs="*", default=None)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--run", default="r1", help="Sub-directory name for this run.")
    parser.add_argument("--out", default="../.eval")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Topics in flight at once. Provider rate limits are per-provider and shared.",
    )
    args = parser.parse_args()

    config = load_config()
    object.__setattr__(config.retrieval.layers, "llm", True)

    if args.model:
        # Keep the configured fallbacks behind the override so a sweep over models still
        # produces a path when one of them is unavailable.
        for role, chain in list(config.llm.roles.items()):
            rest = [m for m in (chain if isinstance(chain, list) else [chain]) if m != args.model]
            config.llm.roles[role] = [args.model, *rest[:2]]

    model_label = (args.model or config.llm.model_for("curriculum_planning") or "none").replace(
        "/", "-"
    )
    topics = args.topics or DEFAULT_TOPICS
    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(topic: str) -> TopicRun:
        async with semaphore:
            run = await run_topic(topic, args.strategy, config, args.budget)
            status = "FAIL" if run.error else f"{run.step_count} steps"
            print(f"  [{args.strategy}/{model_label}] {topic}: {status} ({run.seconds:.0f}s)")
            return run

    print(f"Running {len(topics)} topics — strategy={args.strategy} model={model_label}")
    runs = list(await asyncio.gather(*(guarded(t) for t in topics)))

    out_dir = Path(__file__).resolve().parents[1] / args.out / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.strategy}__{model_label}"
    (out_dir / f"{stem}.md").write_text(render(args.strategy, model_label, runs), encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"strategy": args.strategy, "model": model_label,
                    "runs": [asdict(r) for r in runs]}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir / f'{stem}.md'}")


if __name__ == "__main__":
    asyncio.run(main())
