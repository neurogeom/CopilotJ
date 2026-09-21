# Add cross-process mutual exclusion to sync_agent_configs

- **What:** Add a file lock (e.g. `fcntl.flock`) and an atomic baseline swap (`os.replace` from a temp name) around the refresh + baseline-write critical section in `copilotj/multiagent/agent_loader.py::sync_agent_configs`.
- **Why:** The seed-baseline read → per-file backup/copy → baseline-write sequence is not atomic, and the module-level `_synced` flag only guards within one process. Two processes sharing one `$COPILOTJ_HOME` (not the default single-bridge deployment, but plausible under multi-process or test setups) could both refresh concurrently. The backup logic itself is data-safe under concurrent syncers (the collision-avoiding `.bak.YYYYMMDD(.N)` naming + `filecmp` short-circuit preserve every distinct state), so this is hardening, not a known data-loss path.
- **Pros:** Eliminates the one identified concurrency fragility in the agents refresh; makes multi-instance `$COPILOTJ_HOME` safe if ever introduced.
- **Cons:** Adds locking complexity for a scenario (multi-process sharing) that the single-bridge deployment doesn't hit; risk of deadlocks if not careful, and Windows `fcntl` portability needs a fallback.
- **Context:** Surfaced by the `/review` adversarial pass (Claude F1, 2026-07-02); deferred as hardening. Single bridge server per `$COPILOTJ_HOME` is the supported deployment, so real-world concurrency is unlikely. Reference: `sync_agent_configs`, `_save_seed_baseline` in `copilotj/multiagent/agent_loader.py`.
- **Depends on / blocked by:** Nothing.
