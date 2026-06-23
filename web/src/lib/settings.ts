/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { getModelCapabilities } from "../apis";
import { isUseServer } from "../apis";
import type { ExplicitModel, ThreadConfigModel } from "../apis";

/** The persisted VLM slot. Structurally identical to ConfigData["vlm"]; kept
 * local to avoid a store ↔ composable type cycle. */
export interface VlmConfig {
  model: string | null;
  api_key: string | null;
  base_url: string | null;
  provider: string | null;
  useMainModel: boolean;
}

/** Resolve the concrete model name a {use_server:true} or explicit choice maps
 * to, for capability lookups. Returns null when nothing is configured. */
export function resolveMainModelName(
  choice: ThreadConfigModel | null,
  serverModel: ExplicitModel | null,
): string | null {
  if (!choice) return null;
  return isUseServer(choice) ? (serverModel?.name ?? null) : choice.name;
}

/** A separate, dedicated vision model is configured (i.e. not "use main model"). */
export function configuredSeparateVlm(vlm: VlmConfig): boolean {
  return !vlm.useMainModel && !!vlm.model;
}

export type VisionResolvableReason = "main-no-vision";

export interface VisionResolvableResult {
  ok: boolean;
  reason?: VisionResolvableReason;
}

/**
 * Whether Vision can actually work, mirroring the backend's
 * `vision_available = llm_supports_vision OR vlm_configured`
 * (copilotj/core/config.py). Vision is resolvable when it is disabled, when a
 * separate VLM is configured, or when the main model supports image input.
 *
 * Only an AUTHORITATIVE "main model lacks vision" result blocks (ok:false): the
 * model must be found in the LiteLLM catalog (ModelCapabilities.source ===
 * "litellm_db") with supports_vision === false. Unknown models (source ===
 * "unknown", e.g. "invalidmodel") and heuristic guesses (source ===
 * "heuristic") are ALLOWED — we can't confirm Vision is broken, so we never
 * block on uncertainty. (An empty required Model field is rejected separately,
 * at submit time, in Settings.vue.)
 *
 * `mainModelName` should already be resolved (see resolveMainModelName); pass
 * null when no model is chosen.
 */
export async function checkVisionResolvable(opts: {
  visionEnabled: boolean;
  mainModelName: string | null;
  vlm: VlmConfig;
}): Promise<VisionResolvableResult> {
  if (!opts.visionEnabled) return { ok: true };
  if (configuredSeparateVlm(opts.vlm)) return { ok: true };
  if (!opts.mainModelName) return { ok: true };
  try {
    const caps = await getModelCapabilities(opts.mainModelName);
    if (caps.supports_vision) return { ok: true };
    // Only the authoritative catalog can confirm a model lacks vision. Unknown
    // / heuristic results get the benefit of the doubt — allow.
    return caps.source === "litellm_db" ? { ok: false, reason: "main-no-vision" } : { ok: true };
  } catch {
    // Lookup failed — can't confirm Vision is broken, so allow.
    return { ok: true };
  }
}

// ---------------------------------------------------------------------------
// Unified validation — the single source of truth shared by the Settings
// dialog and the first-run Wizard. Every rule lives here as a pure predicate
// returning a ValidationConcern; both parents call these on their own draft so
// the rule can never drift between the two surfaces. `key` decouples the module
// from either UI's tab/step model — each consumer maps a key to its own
// tab (Settings) or step (Wizard).
// ---------------------------------------------------------------------------

export type ValidationKey = "agreement" | "base" | "model" | "vlm" | "vision";

export interface ValidationConcern {
  key: ValidationKey;
  ok: boolean;
  /** Human-facing message; empty when `ok`. Ported here so both surfaces stay identical. */
  message: string;
}

/** The normalized shape both the Settings draft and the Wizard reactive satisfy. */
export interface ValidationInput {
  userAgreement: boolean;
  visionEnabled: boolean;
  model: ThreadConfigModel | null;
  vlm: VlmConfig;
  /** Owned by SettingsConnection; `"ok"` only after a successful connect (and reset to
   * `"idle"` when the URL is later edited — see SettingsConnection). */
  connectionStatus: "idle" | "testing" | "ok" | "fail";
  apiBaseUrl: string;
}

/** Map a parent's camelCase VLM draft onto the snake_case VlmConfig this module uses. */
export function toVlmConfig(v: {
  useMainModel: boolean;
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  provider: string | null;
}): VlmConfig {
  return {
    model: v.model,
    api_key: v.apiKey,
    base_url: v.baseUrl,
    provider: v.provider,
    useMainModel: v.useMainModel,
  };
}

/** Usage agreement accepted (first-run gate; Settings does not re-enforce this). */
export function validateAgreement(i: ValidationInput): ValidationConcern {
  const ok = i.userAgreement;
  return { key: "agreement", ok, message: ok ? "" : "Please accept the usage notice to continue." };
}

/** The server URL is non-empty and currently connected. */
export function validateBase(i: ValidationInput): ValidationConcern {
  const ok = i.apiBaseUrl.trim() !== "" && i.connectionStatus === "ok";
  return { key: "base", ok, message: ok ? "" : "Please test the server URL on the Base tab before saving." };
}

/** A model is chosen: the server default, or a non-empty explicit name. */
export function validateModel(i: ValidationInput): ValidationConcern {
  const m = i.model;
  const ok = !!m && (isUseServer(m) || !!m.name);
  return {
    key: "model",
    ok,
    message: ok ? "" : `Please choose a model on the Model tab, or enable “Use Default Model”.`,
  };
}

/** When Vision is on, either reuse the main model or pick a separate VLM. */
export function validateVlm(i: ValidationInput): ValidationConcern {
  const ok = !i.visionEnabled || i.vlm.useMainModel || !!i.vlm.model;
  return {
    key: "vlm",
    ok,
    message: ok ? "" : `Vision is enabled — choose a vision model on the Vision tab, or enable “Use main model”.`,
  };
}

/**
 * Vision can actually run (async — probes model capabilities). Only a confirmed
 * "main model lacks vision" result blocks; unknown/heuristic models and lookup
 * failures are allowed (delegated to checkVisionResolvable). `mainModelName`
 * should already be resolved (see resolveMainModelName).
 */
export async function validateVision(i: ValidationInput, mainModelName: string | null): Promise<ValidationConcern> {
  if (!i.visionEnabled) return { key: "vision", ok: true, message: "" };
  const result = await checkVisionResolvable({
    visionEnabled: true,
    mainModelName,
    vlm: i.vlm,
  });
  return {
    key: "vision",
    ok: result.ok,
    message: result.ok
      ? ""
      : `The selected model "${mainModelName}" does not support image input, and no separate vision model is configured. Configure one on this tab, or disable Vision on the Notice tab.`,
  };
}
