# Eval harness

Scores the whole memory loop against gold labels: facts go in through the real
Writer write path, the Editor pipeline runs over them, and questions go through
the real retrieval. It answers "were the merges right?", not just "how many
merges happened?".

| What | Measured as |
|---|---|
| Dedup | Pairwise precision/recall over facts: two facts are merged when they end up with the same surviving memory (Writer skip or Editor `CANONICAL`) |
| Contradictions | Precision of conflict edges; recall over outdated facts; share of outdated facts actually superseded; current facts wrongly superseded |
| Entity resolution | Pairwise precision/recall over entity names; wrong-merge rate = `MERGED_INTO` edges joining different real entities |
| Retrieval | recall@5, MRR, and how often an outdated fact outranks the current one; measured before and after the Editor |
| Cost | LLM calls, tokens, list-price cost, and LLM time per Editor step |

Results for each stage of the work are in `results/` (`01-baseline.json` to
`08-final.json`, plus `model-haiku-4-5.json`); the summary table is in the
top-level README. Only the final run's recordings are kept in `cache/`, so the
earlier stages are a record, not something CI can replay.

## Running it

```bash
docker compose -f eval/docker-compose.yml up -d     # scratch Neo4j on :7688
(cd extensions/dual-memory && npm ci)
cd editor && pip install -e ".[dev]"

python -m editor.evaluation              # replay recorded LLM/embedding calls, no keys
python -m editor.evaluation --check      # also fail if a metric misses thresholds.json
```

The harness wipes its database on every run. It refuses to touch a non-empty
database it didn't create (no `:EvalSentinel` node), and it reads its connection
only from `EVAL_NEO4J_URI` / `EVAL_NEO4J_USER` / `EVAL_NEO4J_PASSWORD`
(default `bolt://localhost:7688`), never from `editor/.env`.

## Record and replay

Every LLM reply and embedding is stored in `cache/` keyed by a hash of the full
request (model, effort, prompts). Replay sends byte-identical prompts or fails
with a cache miss, which is how CI notices that an Editor prompt changed. After
changing a prompt, threshold, or the fixture, re-record locally:

```bash
ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... python -m editor.evaluation --record --prune
```

`--prune` drops recordings the run no longer used. `--model` and `--effort`
evaluate other Editor settings; each has its own recordings.

## The fixture

`fixture.json` holds 40 synthetic sessions (January to September 2026) of the
facts a Writer would extract about a fictional user, Sam Ortiz. It stands in for
the Writer's LLM extraction, so extraction quality is not measured here.

- **Facts** carry a `cluster`. Facts sharing one are the same fact restated;
  every other fact is its own cluster (its id). Named single-fact clusters
  (`streak-50`) just make references readable.
- **Chains** list one attribute's values over time (`lives-seattle` →
  `lives-portland`). Each later value makes the earlier ones outdated. A chain's
  `older` facts are only true before the change ("Chris Lee is User's teammate at
  Northwind"); its `newer` facts imply the latest value without stating it
  ("User is building a pipeline at Contoso"). Linking the old side to the new
  side counts as correct; recall counts only the `sequence`.
- **ignore_pairs** are pairs where flagging a conflict is defensible either way
  (training for a race vs. having finished it). They count neither way.
- **Entities** map every name to a `gold` identity. The table includes the traps
  that motivated three-tier resolution: two people aliased "Chris", "Jordan" the
  person and the country, "Portland" and "Portland, Maine", "Mei" (the sister)
  and "Mei Lin" (a coworker).
- **Questions** give answer groups of clusters; a group is satisfied when any of
  its clusters is retrieved. `current` questions also list the outdated clusters,
  to measure whether stale facts outrank current ones.

The loader validates references, so a typo in a cluster name fails loudly.

## External benchmark: LoCoMo

The fixture above was written by the same person who tuned the thresholds, so
it can't be the only number. `python -m editor.evaluation.locomo` runs
[LoCoMo](https://github.com/snap-research/locomo) (10 conversations between
two people, 272 dated sessions, 5,882 turns) through the shipped code:

1. each session goes through the Writer's real LLM extraction (Sonnet 5) and
   `writeFacts`, stamped with the session date;
2. the Editor runs over each conversation's graph (Sonnet 5, medium effort);
3. each question goes through `Retrieval.retrieveContext()`, i.e. the exact
   `<graph-memories>` block the plugin injects (8 memories);
4. a fixed reader (Sonnet 5, low effort) answers from that block alone, and a
   judge (Haiku 4.5) marks it CORRECT or WRONG against the gold answer.

Category 5 (adversarial, wrong-premise questions) is excluded, as in prior
LoCoMo evaluations. Result (`results/locomo.json`), one run, $17.03 in API calls:

| Category | Questions | Judge accuracy | Token F1 | Evidence session among recalled memories |
|---|---|---|---|---|
| single-hop | 841 | 56.8% | 0.412 | 77.5% |
| multi-hop | 282 | 33.0% | 0.319 | 88.6% |
| open-domain | 96 | 36.5% | 0.217 | 66.3% |
| temporal | 321 | 8.1% | 0.086 | 77.0% |
| **overall** | **1,540** | **41.0%** | **0.315** | **78.8%** |

What it says about the system, not the benchmark:

- **Extraction is the bottleneck, not retrieval.** For 79% of questions a
  recalled memory came from a session holding the evidence, but the specific
  detail was never written: the Writer targets 3-6 durable facts per
  conversation (about 6 per 22-turn session here), which suits an assistant's
  long-term memory and loses LoCoMo's conversational detail.
- **Temporal questions fail because the Writer never sees a date.** Facts say
  "last Saturday" or nothing; 9 of 1,582 extracted facts carried an event date.
  Passing the session date into extraction and showing memory dates in the
  injected context is the obvious fix and would need its own measured run.
- It found two more bugs: a partial date from the LLM ("--08-15") aborted a
  whole session's write, and judge replies with prose after the JSON crashed
  parsing. Both are fixed.

Numbers aren't comparable across papers unless the reader and judge match;
this is a like-for-like baseline for later changes to this system. The
dataset, recordings, and failure examples stay local (`eval/locomo/`,
gitignored).

## Label corrections

Gold labels changed after results were seen. Each fixes a labelling gap under
the rules above rather than accommodating a model answer:

- `s40.3` "Chris Lee left Northwind to join a startup" outdates `s06.1` "Chris
  Lee is User's teammate on Northwind's routing service": added chain
  `chris-lee-team`.
- `s17.3` "User will relocate to Portland for the Contoso job" implies the new
  employer: added to the `employer` chain's `newer` facts.
- A relocation plan superseded by the move itself, and an interview superseded
  by a plan that assumes the job, are defensible either way (like the existing
  "rewriting in Rust" / "written in Rust" pair): added to `ignore_pairs`.

## Limitations

- One fixture, written by the same person who tuned the thresholds. The entity
  and conflict-candidate thresholds were chosen by looking at its similarity
  distributions; there is no held-out split yet. A public benchmark
  (LongMemEval, LoCoMo) would give an external number.
- 102 facts is small. Precision on a handful of predicted pairs moves in large
  steps.
- Extraction isn't scored: the fixture starts from extracted facts.
