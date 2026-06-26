# Bootstrap frontend testing

**What:** Add a frontend test framework to `web/` (vitest + Vue Test Utils + jsdom), wire a `just test-web` command, and seed it with tests for the Settings base-URL lock + reconnect feature.

**Why:** `web/` currently has zero automated tests (`just test` runs Python doctests only). The Settings base-URL lock/reconnect has real edge cases that only get manual coverage today: the pre-start window where `status === "init"` but `loading`/`optimizing` is true; `/api/ping` succeeding while `/api/config` fails; the `error`/`aborted` terminal states keeping the field locked.

**Pros:** Catches regressions in the lock condition and reconnect failure handling; gives the whole frontend a safety net; matches the project's "well-tested code is non-negotiable" bar.

**Cons:** Introduces test-framework scope (config, jsdom, CI wiring) broader than any single feature; first-time setup cost.

**Context:** Surfaced by the `/plan-eng-review` of the Settings base-URL lock change on `settings-vision-nav`. The lock lives in `web/src/components/Settings.vue` (`threadActive` computed + `onSubmit` guard + `onReconnect`); the reconnect logic is `web/src/store/config.ts` `applyServerConfig()`; the lock signal is `web/src/store/thread.ts` `useActiveThread().thread` (`status`, `loading`, `optimizing`). Review decision: ship the feature on manual verification and defer the framework to its own piece of work. Seed tests should cover: `threadActive` across `init`/`loading`/`optimizing`/`started`/`error`/`aborted`; the `onSubmit` base-tab guard; `applyServerConfig` success vs `/api/config` failure (returns false, no mutation); `onReconnect` close-on-success / warn-on-failure.

**Depends on / blocked by:** Nothing.
