export type Paper = {
  id: string;
  title: string;
  abstract: string | null;
  year: number | null;
  authors: string[];
  venue: string | null;
  citation_count: number | null;
  pdf_url: string | null;
  landing_url: string | null;
  external_ids: string[];
  depth: string;
  has_abstract: boolean;
  found_by: string[];
  score: number;
};

/**
 * §5's four questions. `source` distinguishes a reasoned explanation from one derived
 * from the citation graph — the UI must never present the second as the first.
 */
export type Explanation = {
  why_it_matters: string;
  what_it_assumes: string;
  what_it_teaches: string;
  why_for_you: string;
  source: "llm" | "structural";
  asserted_by: string;
};

/** The structural evidence behind a paper's placement, shown rather than hidden. */
export type Signals = {
  co_citations: number;
  pagerank: number;
  age_rescaled_pagerank: number;
  in_degree: number;
  out_degree: number;
  discovered_by_expansion: boolean;
};

export type PathStep = {
  order: number;
  level: number;
  paper: Paper;
  role: string;
  signals: Signals;
  explanation: Explanation;
  subtopic_id: string | null;
  prerequisite_ids: string[];
  already_read: boolean;
};

export type Subtopic = {
  id: string;
  label: string;
  summary: string | null;
  order: number;
  named_by_llm: boolean;
};

export type PathEdge = {
  prerequisite_id: string;
  dependent_id: string;
  source: "citation" | "co_citation" | "llm_judgment";
  confidence: number | null;
  reason: string | null;
  asserted_by: string;
};

export type PathResponse = {
  topic: string;
  count: number;
  levels: number;
  steps: PathStep[];
  subtopics: Subtopic[];
  edges: PathEdge[];
  layers_used: string[];
  stages_run: string[];
  degraded: boolean;
  /** How much the system believes its own answer, 0-1, with the reasons. */
  confidence: number;
  confidence_reasons: string[];
  notes: string[];
};

export class ApiError extends Error {}

export async function fetchPath(
  topic: string,
  signal?: AbortSignal,
): Promise<PathResponse> {
  const url = `/api/path?topic=${encodeURIComponent(topic)}&limit=20`;
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (cause) {
    if ((cause as Error).name === "AbortError") throw cause;
    throw new ApiError(
      "Can't reach the PaperThread API. Start it with: uvicorn paperthread.api.main:app --reload",
    );
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(detail || `Building the path failed with status ${response.status}.`);
  }
  return (await response.json()) as PathResponse;
}

/** All five layers in pipeline order, so the strip shows what is OFF as well as ON. */
export const ALL_LAYERS = [
  { key: "lexical", label: "Lexical", tier: "L0" },
  { key: "local_nlp", label: "Local NLP", tier: "L1" },
  { key: "embeddings", label: "Embeddings", tier: "L2" },
  { key: "reranking", label: "Rerank", tier: "L3" },
  { key: "llm", label: "LLM judgment", tier: "L4" },
] as const;

/**
 * The pipeline stages, so the UI can show which ran. Stage 0 (topic decomposition before
 * search) and stage 6 (personalization) are listed because they are part of the product
 * and are NOT built — an absent capability the user cannot see is one they will assume.
 */
export const ALL_STAGES = [
  { key: "decompose", label: "Decompose", built: false },
  { key: "retrieve", label: "Retrieve", built: true },
  { key: "expand", label: "Expand graph", built: true },
  { key: "score", label: "Score", built: true },
  { key: "judge", label: "Judge", built: true },
  { key: "order", label: "Order", built: true },
  { key: "personalize", label: "Personalize", built: false },
] as const;

export const ROLE_LABELS: Record<string, string> = {
  foundation: "foundation",
  breakthrough: "breakthrough",
  alternative: "alternative",
  extension: "extension",
  critique: "critique",
  survey: "survey",
  application: "application",
  unclassified: "unclassified",
};

/**
 * The three acts of a planned path, mirroring `STAGE_*` in
 * `backend/paperthread/retrieval/curriculum.py`. A planned path arrives as one paper per
 * level, so levels carry no grouping — the act does. Keys must match the backend exactly.
 */
export const ACTS = [
  {
    key: "prerequisite",
    label: "Before the paper",
    note: "Groundwork the main paper takes for granted.",
  },
  {
    key: "anchor",
    label: "The paper",
    note: "The work the topic is named for.",
  },
  {
    key: "followup",
    label: "After the paper",
    note: "What the field built once the idea landed.",
  },
] as const;

export type ActKey = (typeof ACTS)[number]["key"];

const ACT_KEYS = new Set<string>(ACTS.map((act) => act.key));

/** A planned path tags every step with its act; a structural one tags none. */
export function actOf(step: PathStep): ActKey | null {
  return step.subtopic_id && ACT_KEYS.has(step.subtopic_id)
    ? (step.subtopic_id as ActKey)
    : null;
}

/**
 * Whether the prerequisite EDGES were judged, as opposed to inferred from shared citations.
 *
 * Read from the edges themselves rather than from a stage name: the structural pipeline
 * records `judge:llm` while the planned strategies record only `strategy:<name>`, so keying
 * off the stage list reports every planned path as unjudged even though a model asserted
 * every edge in it. `every`, not `some` — one inferred edge and the whole thread is inferred.
 */
export function edgesJudged(result: PathResponse): boolean {
  return (
    result.edges.length > 0 &&
    result.edges.every((edge) => edge.source === "llm_judgment")
  );
}

/** Whether the citation-graph signals on a step were actually measured. */
export function hasMeasuredSignals(step: PathStep): boolean {
  const { co_citations, pagerank, age_rescaled_pagerank, in_degree, out_degree } = step.signals;
  return (
    co_citations > 0 || pagerank !== 0 || age_rescaled_pagerank !== 0 || in_degree > 0 || out_degree > 0
  );
}
