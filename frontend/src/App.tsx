import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ACTS,
  ALL_LAYERS,
  ALL_STAGES,
  ApiError,
  actOf,
  edgesJudged,
  fetchPath,
  hasMeasuredSignals,
  type ActKey,
  type PathResponse,
  type PathStep,
  type Subtopic,
  ROLE_LABELS,
} from "./api";

/** Papers older than this predate most current topic vocabulary — the ancestor band. */
const ANCESTOR_CUTOFF = 2015;

const EXAMPLES = ["diffusion models", "regularization", "transformers"];

export default function App() {
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState<PathResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => () => inFlight.current?.abort(), []);

  const run = useCallback(async (query: string) => {
    const trimmed = query.trim();
    if (!trimmed) return;

    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    setLoading(true);
    setError(null);
    try {
      setResult(await fetchPath(trimmed, controller.signal));
    } catch (cause) {
      if ((cause as Error).name === "AbortError") return;
      setError(cause instanceof ApiError ? cause.message : "Something went wrong.");
      setResult(null);
    } finally {
      if (inFlight.current === controller) setLoading(false);
    }
  }, []);

  const submit = useCallback(
    (event: React.FormEvent) => {
      event.preventDefault();
      void run(topic);
    },
    [run, topic],
  );

  const pick = useCallback(
    (value: string) => {
      setTopic(value);
      void run(value);
    },
    [run],
  );

  const showLanding = !result && !error && !loading;

  return (
    <>
      <StickyBar
        topic={topic}
        setTopic={setTopic}
        onSubmit={submit}
        loading={loading}
        active={!showLanding}
      />

      <header className={`hero ${showLanding ? "" : "hero-compact"}`}>
        <div className="hero-inner">
          <div className="hero-lead">
            <Wordmark />

            {showLanding && (
              <>
                <h1 className="hero-headline">Read a field in the order it was built.</h1>
                <p className="hero-sub">
                  Name a research topic. PaperThread plans a reading path through its
                  literature — the groundwork first, then the paper the topic is named for,
                  then what the field did with it. Every step says why it is there.
                </p>
              </>
            )}

            <SearchForm
              topic={topic}
              setTopic={setTopic}
              onSubmit={submit}
              loading={loading}
              onPick={pick}
            />
          </div>

          {showLanding && <Specimen />}
        </div>
      </header>

      <main className="shell">
        {loading && <Building />}

        {error && (
          <div className="notice notice-error" role="alert">
            <p className="notice-title">That path did not build.</p>
            <p className="notice-body">{error}</p>
            <p className="notice-body">
              Providers rate-limit and time out under load, and a retry often succeeds where
              the first attempt did not.
            </p>
            <button
              type="button"
              className="notice-action"
              onClick={() => void run(topic)}
              disabled={!topic.trim()}
            >
              Try “{topic.trim() || "this topic"}” again
            </button>
          </div>
        )}

        {result && result.steps.length > 0 && <Result result={result} />}

        {result && result.steps.length === 0 && !error && (
          <>
            <div className="notice">
              <p className="notice-title">No path came back for “{result.topic}”.</p>
              <p className="notice-body">
                A provider that failed returns nothing, which is not the same as nothing
                matching — the build log below says which ran. Try a broader topic, or run
                this one again once the providers reset.
              </p>
            </div>
            <BuildLog result={result} />
          </>
        )}

        {showLanding && <Landing onPick={pick} />}
      </main>

      <footer className="footer">
        <p>
          PaperThread orders papers by what they assume, not by what they match. Metadata
          comes from arXiv, Crossref and OpenAlex.
        </p>
      </footer>
    </>
  );
}

function Wordmark({ small = false }: { small?: boolean }) {
  return (
    <p className={`wordmark${small ? " wordmark-small" : ""}`}>
      Paper<span className="wordmark-thread">Thread</span>
    </p>
  );
}

/* ---------------------------------------------------------------- search */

function SearchForm({
  topic,
  setTopic,
  onSubmit,
  loading,
  onPick,
}: {
  topic: string;
  setTopic: (value: string) => void;
  onSubmit: (event: React.FormEvent) => void;
  loading: boolean;
  onPick: (value: string) => void;
}) {
  return (
    <form className="search" onSubmit={onSubmit}>
      <div className="search-row">
        <label className="sr-only" htmlFor="topic">
          Research topic
        </label>
        <input
          id="topic"
          className="search-input"
          value={topic}
          onChange={(event) => setTopic(event.target.value)}
          placeholder="Diffusion models"
          autoComplete="off"
          autoFocus
        />
        <button className="search-button" type="submit" disabled={loading || !topic.trim()}>
          {loading ? "Building…" : "Build the path"}
        </button>
      </div>
      <div className="search-examples">
        <span className="search-examples-label">Try</span>
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            className="chip-button"
            onClick={() => onPick(example)}
            disabled={loading}
          >
            {example}
          </button>
        ))}
      </div>
    </form>
  );
}

/** Slim bar that takes over once the hero has scrolled away. */
function StickyBar({
  topic,
  setTopic,
  onSubmit,
  loading,
  active,
}: {
  topic: string;
  setTopic: (value: string) => void;
  onSubmit: (event: React.FormEvent) => void;
  loading: boolean;
  active: boolean;
}) {
  const [shown, setShown] = useState(false);

  useEffect(() => {
    if (!active) {
      setShown(false);
      return;
    }
    const onScroll = () => setShown(window.scrollY > 220);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [active]);

  return (
    <div className={`stickybar${shown ? " stickybar-shown" : ""}`} aria-hidden={!shown}>
      <div className="stickybar-inner">
        <Wordmark small />
        <form className="stickybar-search" onSubmit={onSubmit}>
          <label className="sr-only" htmlFor="topic-compact">
            Research topic
          </label>
          <input
            id="topic-compact"
            className="stickybar-input"
            value={topic}
            onChange={(event) => setTopic(event.target.value)}
            placeholder="Another topic"
            autoComplete="off"
            tabIndex={shown ? 0 : -1}
          />
          <button
            className="stickybar-button"
            type="submit"
            disabled={loading || !topic.trim()}
            tabIndex={shown ? 0 : -1}
          >
            {loading ? "Building…" : "Build"}
          </button>
        </form>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- landing */

/**
 * The landing page's job is to make one claim credible before the reader spends a minute
 * waiting for a build. The claim is the product: the papers you most need are not the ones
 * that match your words. The specimen below is a real result, and it doubles as a key for
 * reading the thread that a real path renders.
 */
function Specimen() {
  return (
    <aside className="specimen" aria-label="Example path">
      <p className="eyebrow">Ask for “regularization”</p>

      <ol className="specimen-thread">
        <li className="specimen-node specimen-ancestor">
          <span className="specimen-year">1996</span>
          <span className="specimen-paper">Regression Shrinkage and Selection via the Lasso</span>
          <span className="specimen-tag">groundwork</span>
        </li>
        <li className="specimen-node specimen-ancestor">
          <span className="specimen-year">2005</span>
          <span className="specimen-paper">
            Regularization and Variable Selection via the Elastic Net
          </span>
          <span className="specimen-tag">groundwork</span>
        </li>
        <li className="specimen-node specimen-recent">
          <span className="specimen-year">2014</span>
          <span className="specimen-paper">
            Dropout: A Simple Way to Prevent Neural Networks from Overfitting
          </span>
          <span className="specimen-tag">where you were headed</span>
        </li>
      </ol>

      <p className="specimen-note">
        Neither Lasso nor Elastic Net contains a word you typed. They are here because the
        topic's own papers cite them in common.
      </p>

      <p className="specimen-key">
        <span className="key-swatch key-ancestor" aria-hidden="true" /> before 2015
        <span className="key-swatch key-recent" aria-hidden="true" /> 2015 onward
      </p>
    </aside>
  );
}

function Landing({ onPick }: { onPick: (value: string) => void }) {
  return (
    <section className="landing">
      <div className="landing-cards">
        <div className="landing-card">
          <h3 className="landing-card-title">Every step carries its reasoning</h3>
          <p className="landing-card-body">
            Each paper says what it assumes, what it teaches, and why it sits where it does —
            and whether that came from a model or from the citation graph. The two never
            look alike.
          </p>
        </div>
        <div className="landing-card">
          <h3 className="landing-card-title">It tells you when to doubt it</h3>
          <p className="landing-card-body">
            Every path reports its own confidence and what it could not do. A path built on
            a dropped step says so before you start reading it.
          </p>
        </div>
        <div className="landing-card landing-card-quiet">
          <h3 className="landing-card-title">Not built yet</h3>
          <p className="landing-card-body">
            No reading history and no personalization. “Why this suits you” is answered from
            the path's own structure, not from anything you have read.
          </p>
        </div>
      </div>

      <p className="landing-prompt">
        Building a path for a new topic takes about a minute — reference lists arrive one
        provider request at a time. Start with{" "}
        {EXAMPLES.map((example, index) => (
          <span key={example}>
            {index > 0 && (index === EXAMPLES.length - 1 ? ", or " : ", ")}
            <button type="button" className="link-button" onClick={() => onPick(example)}>
              {example}
            </button>
          </span>
        ))}
        .
      </p>
    </section>
  );
}

/* ---------------------------------------------------------------- loading */

/**
 * The first build of a topic takes about a minute — stage 2 fetches reference lists under
 * each provider's rate limit. An elapsed counter is the honest thing to show: the API is a
 * single request, so there is no real per-stage progress to report and inventing one would
 * be a lie told to pass the time.
 */
function Building() {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, []);

  return (
    <section className="building" aria-live="polite">
      <div className="building-head">
        <span className="building-pulse" aria-hidden="true" />
        <p className="building-title">Following citations</p>
        <span className="building-elapsed">{elapsed}s</span>
      </div>
      <p className="building-body">
        Reference lists come in one provider request at a time, under each provider's rate
        limit, so the first build of a topic takes about a minute. Results are cached — the
        next build of this topic will not.
      </p>
      <ol className="skeleton" aria-hidden="true">
        {[0, 1, 2].map((row) => (
          <li key={row} className="skeleton-row">
            <span className="skeleton-dot" />
            <span className="skeleton-lines">
              <span className="skeleton-line skeleton-line-short" />
              <span className="skeleton-line" />
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------------------------------------------------------------- result */

function Result({ result }: { result: PathResponse }) {
  const judged = edgesJudged(result);
  const acts = useMemo(() => groupIntoActs(result.steps), [result.steps]);

  return (
    <>
      <PathHeader result={result} acts={acts} />
      <Confidence result={result} />
      <Gaps result={result} />
      {result.subtopics.length > 0 && <SubtopicLegend subtopics={result.subtopics} />}
      <Path result={result} acts={acts} judged={judged} />
      <BuildLog result={result} />
    </>
  );
}

type ActGroup = { key: ActKey | "level"; label: string; note: string; steps: PathStep[] };

/**
 * A planned path arrives as one paper per level, so `level` carries no grouping at all and
 * rendering a heading per level produces one heading per paper. The act — prerequisite,
 * anchor, follow-up — is the real structure. A structural path has no acts and genuinely
 * multi-paper levels, so it falls back to those.
 */
function groupIntoActs(steps: PathStep[]): ActGroup[] {
  if (steps.some((step) => actOf(step) !== null)) {
    return ACTS.map((act) => ({
      key: act.key,
      label: act.label,
      note: act.note,
      steps: steps.filter((step) => actOf(step) === act.key),
    })).filter((group) => group.steps.length > 0);
  }

  const byLevel = new Map<number, PathStep[]>();
  for (const step of steps) {
    const bucket = byLevel.get(step.level);
    if (bucket) bucket.push(step);
    else byLevel.set(step.level, [step]);
  }
  return [...byLevel.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([level, group]) => ({
      key: "level" as const,
      label: `Level ${level}`,
      note:
        level === 0
          ? "No prerequisite inside this path — start anywhere here."
          : "Depends on papers in earlier levels.",
      steps: group,
    }));
}

function PathHeader({ result, acts }: { result: PathResponse; acts: ActGroup[] }) {
  const years = result.steps.map((step) => step.paper.year).filter((y): y is number => y !== null);
  const span = years.length > 1 ? `${Math.min(...years)}–${Math.max(...years)}` : null;

  // Acts are a real breakdown worth stating; levels are an implementation detail of the
  // ordering, so those get counted rather than enumerated.
  const shape = acts.some((act) => act.key !== "level")
    ? acts.map((act) => `${act.steps.length} ${act.label.toLowerCase()}`).join(" · ")
    : `${result.levels} ${result.levels === 1 ? "level" : "levels"} of dependency`;

  return (
    <section className="pathhead">
      <p className="eyebrow">Reading path</p>
      <h1 className="pathhead-topic">{result.topic}</h1>
      <p className="pathhead-facts">
        <strong>{result.count} papers</strong>
        {span && <> spanning {span}</>} · {shape}
      </p>
    </section>
  );
}

/**
 * How much the path trusts itself, stated before the reader invests in it.
 *
 * Not decoration. A learner reviewing this system found that a path they scored 22/25 and
 * one they scored 9/25 rendered identically — same confident stage labels, same fluent
 * rationales — and the 9/25 one described a video-understanding paper as "the paper that
 * defines the Transformer". When the system is sometimes wrong, the output has to carry
 * the difference, because the reader has no other way to find it.
 */
function Confidence({ result }: { result: PathResponse }) {
  const band =
    result.confidence >= 0.8 ? "high" : result.confidence >= 0.55 ? "medium" : "low";
  const percent = Math.round(result.confidence * 100);
  const headline = {
    high: "This path reaches the topic and every step was verified.",
    medium: "This path is usable, with gaps.",
    low: "Treat this path with caution.",
  }[band];

  return (
    <section className={`confidence confidence-${band}`} aria-label="Path confidence">
      <div className="confidence-main">
        <div className="confidence-head">
          <span className="confidence-band">{band} confidence</span>
          <span className="confidence-value">{percent}%</span>
        </div>
        <div
          className="confidence-meter"
          role="meter"
          aria-valuenow={percent}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Path confidence"
        >
          <span className="confidence-fill" style={{ width: `${percent}%` }} />
        </div>
        <p className="confidence-headline">{headline}</p>
      </div>
      {result.confidence_reasons.length > 0 && (
        <ul className="confidence-reasons">
          {result.confidence_reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

/**
 * The unbuilt capabilities, said once at the top in the reader's language.
 *
 * The full layer and stage detail still renders in the build log below — this exists so
 * that a reader who never scrolls that far still learns that the path is not personalized.
 * An absent capability the reader cannot see is one they will assume exists.
 */
function Gaps({ result }: { result: PathResponse }) {
  const ran = new Set(result.stages_run);
  const missing = ALL_STAGES.filter((stage) => !stage.built).map((stage) => stage.label);
  const layersOff = ALL_LAYERS.filter((layer) => !result.layers_used.includes(layer.key)).length;
  if (missing.length === 0 && layersOff === 0 && !result.degraded) return null;
  void ran;

  return (
    <p className="gaps">
      This path is not tailored to you — reading history and personalization are not built.
      {layersOff > 0 && ` ${layersOff} of ${ALL_LAYERS.length} retrieval layers were inactive.`}{" "}
      <a className="link-button" href="#build-log">
        See how it was built
      </a>
    </p>
  );
}

function Path({
  result,
  acts,
  judged,
}: {
  result: PathResponse;
  acts: ActGroup[];
  judged: boolean;
}) {
  const titles = useMemo(
    () => new Map(result.steps.map((step) => [step.paper.id, step.paper.title])),
    [result.steps],
  );
  const subtopics = useMemo(
    () => new Map(result.subtopics.map((subtopic) => [subtopic.id, subtopic])),
    [result.subtopics],
  );
  const previousId = useMemo(() => {
    const map = new Map<string, string | null>();
    result.steps.forEach((step, index) => {
      map.set(step.paper.id, index === 0 ? null : result.steps[index - 1].paper.id);
    });
    return map;
  }, [result.steps]);
  const expanded = result.stages_run.includes("expand");
  const actful = acts.some((act) => act.key !== "level");

  return (
    <div className={`thread ${judged ? "thread-judged" : "thread-inferred"}`}>
      <p className="thread-legend">
        {judged
          ? "Solid line: every prerequisite here was judged to be a real teaching dependency."
          : "Dashed line: the order is enforced, but whether each step is a real prerequisite has not been judged."}
      </p>

      {acts.map((act) => (
        <section key={act.key + act.label} className={`act act-${act.key}`}>
          <div className="act-head">
            <h2 className="act-label">{act.label}</h2>
            <span className="act-count">
              {act.steps.length} {act.steps.length === 1 ? "paper" : "papers"}
            </span>
            <p className="act-note">{act.note}</p>
          </div>
          <ol className="act-steps">
            {act.steps.map((step) => (
              <StepCard
                key={step.paper.id}
                step={step}
                titles={titles}
                previousId={previousId.get(step.paper.id) ?? null}
                subtopic={step.subtopic_id ? subtopics.get(step.subtopic_id) : undefined}
                showRole={!actful}
                showExpansion={expanded}
              />
            ))}
          </ol>
        </section>
      ))}
    </div>
  );
}

function StepCard({
  step,
  titles,
  previousId,
  subtopic,
  showRole,
  showExpansion,
}: {
  step: PathStep;
  titles: Map<string, string>;
  previousId: string | null;
  subtopic?: Subtopic;
  showRole: boolean;
  showExpansion: boolean;
}) {
  const [open, setOpen] = useState(false);
  const { paper, signals, explanation } = step;
  const era =
    paper.year === null ? "unknown" : paper.year < ANCESTOR_CUTOFF ? "ancestor" : "recent";
  const link = paper.landing_url ?? paper.pdf_url ?? undefined;
  const anchor = step.subtopic_id === "anchor";

  // A prerequisite that is simply the step before this one is already said by the order.
  // Listing it on every card turns the one genuinely informative case — a dependency that
  // reaches further back — into noise.
  const notablePrereqs = step.prerequisite_ids.filter((id) => id !== previousId);
  const measured = hasMeasuredSignals(step);

  return (
    <li className={`node node-${era}${anchor ? " node-anchor" : ""}`}>
      <span className="node-marker" aria-hidden="true" />
      <article className="card">
        <div className="card-meta">
          <span className="card-position">{String(step.order + 1).padStart(2, "0")}</span>
          <span className="card-year">{paper.year ?? "year unknown"}</span>
          {paper.venue && <span className="card-venue">{paper.venue}</span>}
          {paper.citation_count !== null && (
            <span className="card-cites">{paper.citation_count.toLocaleString()} citations</span>
          )}
          {showRole && (
            <span className={`role role-${step.role}`}>
              {ROLE_LABELS[step.role] ?? step.role}
            </span>
          )}
          {subtopic && <span className="card-subtopic">{subtopic.label}</span>}
        </div>

        <h3 className="card-title">
          {link ? (
            <a href={link} target="_blank" rel="noreferrer">
              {paper.title}
            </a>
          ) : (
            paper.title
          )}
        </h3>

        {paper.authors.length > 0 && (
          <p className="card-authors">
            {paper.authors.slice(0, 4).join(", ")}
            {paper.authors.length > 4 && ` and ${paper.authors.length - 4} more`}
          </p>
        )}

        {notablePrereqs.length > 0 && (
          <p className="card-prereq">
            <span className="card-prereq-label">Read first</span>
            {notablePrereqs.map((id) => (
              <span key={id} className="card-prereq-item">
                {titles.get(id) ?? id}
              </span>
            ))}
          </p>
        )}

        <Explain explanation={explanation} />

        <div className="card-foot">
          <div className="card-signals">
            {showExpansion && signals.discovered_by_expansion && (
              <span
                className="signal signal-note"
                title="Reached by following citations backwards. Keyword search did not return it."
              >
                found by citation, not search
              </span>
            )}
            {measured && signals.co_citations > 0 && (
              <span
                className="signal"
                title="How many of this topic's own candidate papers cite it"
              >
                cited by {signals.co_citations} of the topic's papers
              </span>
            )}
            {measured && signals.age_rescaled_pagerank !== 0 && (
              <span
                className="signal"
                title="PageRank relative to papers of a similar age, within this topic's subgraph"
              >
                age-rescaled {signals.age_rescaled_pagerank >= 0 ? "+" : ""}
                {signals.age_rescaled_pagerank.toFixed(2)}
              </span>
            )}
            {paper.found_by.length > 0 && (
              <span className="signal" title="Which providers returned this paper">
                via {paper.found_by.join(", ")}
              </span>
            )}
            {!paper.has_abstract && (
              <span
                className="signal signal-gap"
                title="No abstract available. Kept deliberately — coverage is worst for older papers."
              >
                no abstract on file
              </span>
            )}
          </div>

          {paper.has_abstract && (
            <button
              type="button"
              className="card-toggle"
              onClick={() => setOpen((value) => !value)}
              aria-expanded={open}
            >
              {open ? "Hide abstract" : "Read abstract"}
            </button>
          )}
        </div>

        {open && paper.has_abstract && <p className="card-abstract">{paper.abstract}</p>}
      </article>
    </li>
  );
}

/**
 * §5's four questions. The source badge is not a footnote: a structural explanation is
 * measured from the citation graph and a reasoned one is not, and presenting them
 * identically would be the single most misleading thing this UI could do.
 */
function Explain({ explanation }: { explanation: PathStep["explanation"] }) {
  const structural = explanation.source === "structural";
  return (
    <div className={`explain ${structural ? "explain-structural" : "explain-llm"}`}>
      <dl className="explain-grid">
        <dt>Why it matters</dt>
        <dd>{explanation.why_it_matters}</dd>
        <dt>What it assumes</dt>
        <dd>{explanation.what_it_assumes}</dd>
        <dt>What it teaches</dt>
        <dd>{explanation.what_it_teaches}</dd>
        <dt>Why for you</dt>
        <dd>{explanation.why_for_you}</dd>
      </dl>
      <p className="explain-foot">
        <span className="explain-badge">
          {structural ? "Measured, not reasoned" : "Reasoned"}
        </span>
        <span className="explain-by" title="provider / model / prompt version">
          {explanation.asserted_by}
        </span>
      </p>
    </div>
  );
}

function SubtopicLegend({ subtopics }: { subtopics: Subtopic[] }) {
  const named = subtopics.some((subtopic) => subtopic.named_by_llm);
  return (
    <section className="subtopics" aria-label="Lines of work">
      <div className="subtopics-head">
        <h2 className="subtopics-title">Lines of work</h2>
        <p className="subtopics-hint">
          {named
            ? "Named by L4, grouped by citation community."
            : "Grouped by citation community — naming needs L4."}
        </p>
      </div>
      <ol className="subtopics-list">
        {subtopics.map((subtopic) => (
          <li key={subtopic.id} className="subtopic">
            <span className="subtopic-mark">{subtopic.order}</span>
            <div>
              <span className="subtopic-label">{subtopic.label}</span>
              {subtopic.summary && <p className="subtopic-summary">{subtopic.summary}</p>}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ---------------------------------------------------------------- build log */

/**
 * Which notes are warnings, per note rather than per response.
 *
 * `degraded` is true whenever the pipeline ran below full capability — which, with L4 off,
 * is always. Colouring every note as an alert on that basis makes "expansion added 665
 * papers" look like a failure and, worse, makes the notes that ARE failures invisible by
 * being indistinguishable from the rest.
 */
function isWarning(note: string): boolean {
  return /degraded|failed|could not|no provider|disabled|inactive|not active|dropped/i.test(note);
}

/**
 * The honesty contract, in full. It renders every layer including the ones that are OFF and
 * every stage including the ones that were never built, because a capability the reader
 * cannot see is one they will assume exists. It sits below the path rather than above it —
 * the reader came for the reading list, not the pipeline — but nothing here is hidden or
 * collapsed, and the summary above links straight to it.
 */
function BuildLog({ result }: { result: PathResponse }) {
  const activeLayers = new Set(result.layers_used);
  const stages = new Set(result.stages_run);
  const judged = edgesJudged(result);
  const expanded = result.steps.filter((step) => step.signals.discovered_by_expansion).length;
  const strategy = result.stages_run
    .find((stage) => stage.startsWith("strategy:"))
    ?.replace("strategy:", "");

  return (
    <section className="buildlog" id="build-log" aria-label="How this path was built">
      <h2 className="buildlog-title">How this path was built</h2>

      <p className="buildlog-verdict">
        {result.steps.length === 0 ? (
          <>
            <strong>No path was produced.</strong> The layers and stages below are what ran;
            the notes are why it came back empty.
          </>
        ) : judged ? (
          <>
            <strong>Ordered path, prerequisites judged.</strong> {result.count} papers over{" "}
            {result.levels} levels. A model asserted every prerequisite edge in this path and
            gave a reason for each.
          </>
        ) : (
          <>
            <strong>Ordered path, prerequisites inferred.</strong> {result.count} papers over{" "}
            {result.levels} levels
            {expanded > 0 && <>; {expanded} found by following citations rather than by search</>}.
            The order is enforced by the citation constraint, but whether each edge is a real
            prerequisite has not been judged — that needs L4.
          </>
        )}
      </p>

      <div className="buildlog-grid">
        <div className="buildlog-block">
          <h3 className="buildlog-label">Retrieval layers</h3>
          <div className="chips">
            {ALL_LAYERS.map((layer) => (
              <span
                key={layer.key}
                className={`chip ${activeLayers.has(layer.key) ? "chip-on" : "chip-off"}`}
                title={
                  activeLayers.has(layer.key)
                    ? `${layer.tier} active`
                    : `${layer.tier} not active`
                }
              >
                <span className="chip-tier">{layer.tier}</span>
                {layer.label}
              </span>
            ))}
          </div>
        </div>

        <div className="buildlog-block">
          <h3 className="buildlog-label">
            Pipeline stages
            {strategy && <span className="buildlog-strategy">strategy: {strategy}</span>}
          </h3>
          <div className="chips">
            {ALL_STAGES.map((stage) => {
              const ran = stages.has(stage.key);
              const className = !stage.built
                ? "chip chip-unbuilt"
                : ran
                  ? "chip chip-ran"
                  : "chip chip-skipped";
              return (
                <span
                  key={stage.key}
                  className={className}
                  title={
                    !stage.built
                      ? "Not built in this version"
                      : ran
                        ? "Ran for this topic"
                        : "Did not run for this topic"
                  }
                >
                  {stage.label}
                </span>
              );
            })}
          </div>
        </div>
      </div>

      {result.notes.length > 0 && (
        <ul className="buildlog-notes">
          {result.notes.map((note) => (
            <li key={note} className={isWarning(note) ? "note note-warn" : "note"}>
              {note}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
