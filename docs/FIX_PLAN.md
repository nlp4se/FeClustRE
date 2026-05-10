# FeClustRE Fix Plan

This document lists the repair steps in dependency order. The goal is to make
the repository usable as a replication package first, then improve architecture
and testability.

## Current Verdict

The project is not reliably usable as-is. The codebase contains useful research
logic, but a new user will likely fail during setup or first runtime because
configuration, dependency setup, imports, and data paths are inconsistent.

Think of this like renovating a building: do not start painting the rooms before
checking the foundation, electricity, and plumbing. Here, the foundation is
dependency installation, importability, configuration, and one reproducible
smoke path.

## Phase 1: Make The Repo Installable

1. Convert `requirements.txt` to UTF-8.
   - Why: package managers and editors expect normal text; the current UTF-16
     file is a portability trap.
   - Tradeoff: keep pinned versions for reproducibility, but review platform
     specific packages such as Windows-only entries before claiming cross-platform
     support.

2. Remove unrelated notebook/server packages from runtime dependencies.
   - Why: the current dependency list appears environment-exported, not curated.
   - Better split:
     - `requirements.txt` for runtime
     - `requirements-dev.txt` for tests, notebooks, visualization, and tooling
   - Tradeoff: a curated list takes more thought, but it makes installation much
     faster and failures easier to diagnose.

3. Add an explicit Python version note.
   - Suggested target: Python 3.10 or 3.11, depending on ML package compatibility.
   - Why: ML libraries and torch versions are sensitive to Python versions.

## Phase 2: Fix Importability

1. Fix the post-processor import.
   - Current problem: `services/feature_extraction_service.py` imports
     `feature_post_processor` as a top-level module.
   - Correct direction: use package-local import from `services`.

2. Remove circular or unnecessary imports.
   - Current risk: `services/neo4j_service.py` imports `TaxonomyBuilder`, while
     `TaxonomyBuilder` is constructed with a Neo4j connection elsewhere.
   - Why: circular dependencies make startup fragile and testing harder.

3. Add a minimal import smoke test.
   - Example coverage:
     - `import app` should not load all models.
     - service modules should import independently.
   - Tradeoff: this requires changing startup design first if imports currently
     perform heavy work.

## Phase 3: Fix Local Configuration

1. Change default Neo4j URI to local Docker.
   - Target default: `bolt://localhost:7687`
   - Keep private/VPN endpoints configurable through environment variables only.

2. Add `.env.example`.
   - Include:
     - `NEO4J_URI`
     - `NEO4J_USER`
     - `NEO4J_PASSWORD`
     - `NEO4J_DATABASE`
     - `OLLAMA_BASE_URL`
     - `OLLAMA_MODEL`
     - `TRANSFEATEX_URL`

3. Make TransfeatEx optional.
   - If `TRANSFEATEX_URL` is missing, health checks should report unavailable
     instead of assuming a private IP.
   - Tradeoff: the `transfeatex` and `hybrid` model modes need clear behavior
     when the service is unavailable: fail fast with a useful message, or skip
     that mode in experiments.

## Phase 4: Fix Application Startup

1. Introduce an app factory.
   - Target shape:
     - `create_app(config_object=None)`
     - register routes
     - initialize lightweight configuration only

2. Lazy-load expensive services.
   - Lazy:
     - T-FREX model
     - sentence-transformer embedding model
     - Neo4j driver
     - Ollama checks
   - Why: `/ping` should work even if ML models or Neo4j are unavailable.

3. Separate health checks by severity.
   - `/ping`: process is alive.
   - `/health`: reports dependency status.
   - Optional future:
     - `/ready`: all required runtime dependencies are available.

## Phase 5: Fix Pipeline Correctness

1. Normalize feature extractor return shapes.
   - Required contract: `extract_features(texts) -> list[list[str]]`
   - Current risk: TransfeatEx returns a flat list, while T-FREX returns a list
     per review.
   - Why: storage code indexes `features_per_review[i]`.

2. Normalize model type names.
   - Current code mixes `t-frex` and `tfrex`.
   - Pick one public API value and support aliases at the boundary.

3. Fix sklearn API usage.
   - Current problem: `affinity='precomputed'` fails in sklearn 1.7.2.
   - Correct modern argument: `metric='precomputed'`.

4. Fix taxonomy root linking.
   - Current risk: generated root IDs do not match the generated node IDs, so the
     app-to-taxonomy relationship may not be created.
   - Add a focused test around `build_and_store_taxonomy`.

5. Fix SciPy linkage distance usage.
   - Review calls that pass full distance matrices into `linkage`.
   - Prefer condensed distance vectors from `pdist`, or use the correct API
     intentionally.

## Phase 6: Fix README And Experiment Scripts

1. Update commands to match actual paths.
   - Current README says `python test_config.py mock`.
   - Actual file: `test/test_config.py`.

2. Replace missing dataset references.
   - Current README and tests reference `ChatGPT.csv`.
   - The file is not present in `data/input/endpoint_1_process_reviews/ai_assistants`.

3. Document exact local startup order.
   - Install dependencies.
   - Start Neo4j.
   - Start Ollama.
   - Start Flask API.
   - Run a small smoke experiment.

4. Add troubleshooting notes.
   - Missing Ollama model.
   - Neo4j authentication failure.
   - First model download delay.
   - TransfeatEx unavailable.

## Phase 7: Add Tests That Matter

Start small. Do not create fake confidence by writing broad tests around broken
architecture.

1. Unit tests:
   - CSV parsing rejects empty/malformed files clearly.
   - preprocessing handles empty and non-string input.
   - feature post-processing is deterministic.
   - clustering handles too few features.

2. Contract tests:
   - every feature extractor mode returns `list[list[str]]`.
   - failed external services produce clear errors.

3. Integration smoke tests:
   - Flask app starts without Neo4j/model downloads for `/ping`.
   - `/health` returns structured unhealthy statuses when dependencies are down.
   - a tiny CSV can run through the local-only path.

4. Optional end-to-end replication test:
   - requires Neo4j and Ollama.
   - should be marked separately because it is slower and infrastructure-heavy.

## Phase 8: Clean Architecture After Usability

Only do this after the repo can install, start, and run a tiny pipeline.

1. Move route handlers out of `app.py`.
   - Suggested structure:
     - `api/routes.py`
     - `services/`
     - `repositories/`
     - `config.py`

2. Separate domain logic from infrastructure.
   - Domain:
     - preprocessing
     - feature extraction contracts
     - clustering
     - taxonomy construction
   - Infrastructure:
     - Neo4j persistence
     - Ollama client
     - TransfeatEx HTTP client

3. Define explicit data contracts.
   - Use dataclasses or typed dictionaries for:
     - review input
     - extracted features
     - clustering candidates
     - taxonomy nodes

4. Replace hidden globals with dependency injection.
   - Why: hidden global services make tests slow, brittle, and order-dependent.

## Recommended Fix Order

Do the work in this order:

1. Keep `.gitignore` fixed.
2. Convert and curate dependency files.
3. Fix imports.
4. Align config with local Docker and `.env.example`.
5. Add app factory and lazy services.
6. Fix feature extractor output contracts.
7. Fix sklearn compatibility.
8. Fix taxonomy root linking.
9. Update README and experiment paths.
10. Add smoke tests.
11. Refactor architecture only after the smoke path is stable.

## Definition Of Usable

The repo becomes usable when a new developer can:

1. Create a clean virtual environment.
2. Install dependencies from UTF-8 requirements files.
3. Start Neo4j locally.
4. Start Ollama locally.
5. Start the Flask API without hidden private-network dependencies.
6. Open `/ping` successfully.
7. Open `/health` and see clear dependency statuses.
8. Upload a tiny CSV and get deterministic clustering candidates or a clear
   actionable error.
9. Run the documented mock experiment with files that actually exist.

That is the bar. Anything below that is still a local research prototype, not a
usable replication package.
