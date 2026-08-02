/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { computed, onMounted, ref, watch, type Ref } from "vue";

import { listModels, listProviderModels, type ModelItem } from "../apis";

export const PROVIDER_OPTIONS = [
  { label: "OpenAI", value: "openai" },
  { label: "Anthropic", value: "anthropic" },
  { label: "Google Gemini", value: "gemini" },
  { label: "Ollama (local)", value: "ollama" },
  { label: "DeepSeek", value: "deepseek" },
  { label: "SiliconFlow", value: "siliconflow" },
  { label: "OpenRouter", value: "openrouter" },
  { label: "OpenAI-compatible", value: "openai-compatible" },
];

/**
 * One-click recommended model presets. Clicking a chip fills the Provider and
 * Model fields together so users skip the manual two-step. Surfaced as chips on
 * the Model and Vision settings tabs (and anywhere else SettingsModel/VLM are
 * reused). The catalog need not list a model — the picker shows a "Current"
 * fallback for any value it does not recognise — so this works even for models
 * the cached LiteLLM DB has not caught up to.
 */
export interface ModelPreset {
  label: string;
  provider: string;
  model: string;
}

export const RECOMMENDED_MODEL_PRESETS: ModelPreset[] = [
  { label: "Claude Opus 4.8", provider: "openrouter", model: "anthropic/claude-opus-4.8" },
  { label: "GPT-5.6 Sol", provider: "openrouter", model: "openai/gpt-5.6-sol" },
];

/**
 * Default API base URL for OpenAI-compatible providers whose endpoint is fixed
 * and publicly known (DeepSeek / SiliconFlow / OpenRouter) plus Ollama's local
 * host. Cloud providers whose SDK picks its own endpoint (OpenAI / Anthropic /
 * Gemini) are intentionally absent — getDefaultBaseUrl returns "" for them so
 * the field stays blank unless overridden.
 */
const PROVIDER_BASE_URLS: Record<string, string> = {
  ollama: "http://localhost:11434",
  deepseek: "https://api.deepseek.com",
  siliconflow: "https://api.siliconflow.cn/v1",
  openrouter: "https://openrouter.ai/api/v1",
};

/** Default API base URL for *provider*, or "" when the SDK chooses its own. */
export function getDefaultBaseUrl(provider: string): string {
  return PROVIDER_BASE_URLS[provider] ?? "";
}

/**
 * Resolve the base URL to display: the explicitly stored value, or the
 * provider's default when none was configured — so the field always reflects
 * the effective endpoint.
 */
export function resolveBaseUrl(provider: string, baseUrl: string | null | undefined): string {
  const stored = (baseUrl ?? "").trim();
  return stored || getDefaultBaseUrl(provider);
}

/** Whether *baseUrl* is a non-empty value that differs from the provider default. */
export function hasCustomBaseUrl(provider: string, baseUrl: string | null | undefined): boolean {
  const stored = (baseUrl ?? "").trim();
  return !!stored && stored !== getDefaultBaseUrl(provider);
}

/**
 * Return the base URL to persist, or null when it is empty or equal to the
 * provider default — only genuine overrides are stored.
 */
export function persistBaseUrl(provider: string, baseUrl: string): string | null {
  const trimmed = (baseUrl ?? "").trim();
  if (!trimmed || trimmed === getDefaultBaseUrl(provider)) return null;
  return trimmed;
}

interface ModelGroup {
  label: string;
  provider: string | null;
  items: { label: string; value: string }[];
}

/** Map a provider value to its display label (reuses PROVIDER_OPTIONS). */
const PROVIDER_LABELS: Record<string, string> = Object.fromEntries(PROVIDER_OPTIONS.map((o) => [o.value, o.label]));

/**
 * Infer a provider string from a model name, matching the backend's prefix-based detection.
 * Used when loading configs that lack an explicit provider (e.g. server auto-detection).
 */
export function inferProvider(modelName: string): string {
  if (!modelName) return "openai-compatible";
  if (modelName.startsWith("ollama/")) return "ollama";
  if (modelName.startsWith("deepseek/")) return "deepseek";
  if (modelName.startsWith("gemini-")) return "gemini";
  if (modelName.startsWith("claude-")) return "anthropic";
  if (modelName.startsWith("gpt-")) return "openai";
  if (modelName.startsWith("zai-org/") || modelName.startsWith("Pro/")) return "siliconflow";
  return "openai-compatible";
}

/** Convert server model items into dropdown {label, value} entries. */
function toItems(provider: string, models: ModelItem[]): { label: string; value: string }[] {
  return models.map((m) => ({
    label: m.label,
    value: provider === "ollama" ? `ollama/${m.id}` : m.id,
  }));
}

/**
 * Reactive model groups for the model picker, sourced from the server
 * (`/api/models`). Cloud providers come from the cached catalog; Ollama is
 * queried live whenever the active provider is `ollama`, using the supplied
 * base URL (default localhost).
 *
 * @param currentModel  the currently selected model value (for the "Current" fallback)
 * @param provider      the active provider (drives which groups are shown)
 * @param ollamaBaseUrl optional Ollama host (only relevant when provider is `ollama`)
 */
export function useModelGroups(
  currentModel: Ref<string>,
  provider: Ref<string>,
  ollamaBaseUrl?: Ref<string | undefined>,
) {
  // provider value -> dropdown items (catalog providers + ollama)
  const itemsByProvider = ref<Record<string, { label: string; value: string }[]>>({});
  let ollamaTimer: ReturnType<typeof setTimeout> | undefined;
  // Live status of the Ollama fetch, surfaced to the UI so failures are not
  // silent: shows a loading spinner / "unreachable" / "reachable but empty".
  const ollamaLoading = ref(false);
  const ollamaSource = ref<string | undefined>();

  async function loadCatalogProviders() {
    try {
      const data = await listModels();
      const next: Record<string, { label: string; value: string }[]> = {};
      for (const [p, pm] of Object.entries(data.providers)) {
        if (p === "ollama") continue;
        next[p] = toItems(p, pm.models);
      }
      itemsByProvider.value = { ...itemsByProvider.value, ...next };
    } catch {
      // Server unreachable — leave groups empty; "Current" fallback still applies.
    }
  }

  async function loadOllama() {
    if (provider.value !== "ollama") return;
    ollamaLoading.value = true;
    let items: { label: string; value: string }[] = [];
    try {
      const data = await listProviderModels("ollama", ollamaBaseUrl?.value);
      ollamaSource.value = data.source;
      items = data.source === "live" ? toItems("ollama", data.models) : [];
    } catch {
      ollamaSource.value = "unreachable";
      items = [];
    } finally {
      ollamaLoading.value = false;
    }
    itemsByProvider.value = { ...itemsByProvider.value, ollama: items };
  }

  onMounted(loadCatalogProviders);

  // Ollama models depend on the host (and only matter for the ollama provider).
  // A short debounce avoids firing a request on every keystroke of the base URL.
  watch(
    [() => provider.value, () => ollamaBaseUrl?.value],
    () => {
      if (provider.value !== "ollama") return;
      if (ollamaTimer) clearTimeout(ollamaTimer);
      ollamaTimer = setTimeout(loadOllama, 350);
    },
    { immediate: true },
  );

  /** Re-fetch Ollama models immediately, bypassing the debounce (e.g. on blur). */
  function reloadOllama() {
    if (provider.value !== "ollama") return;
    if (ollamaTimer) clearTimeout(ollamaTimer);
    void loadOllama();
  }

  const modelGroups = computed<ModelGroup[]>(() => {
    const groups: ModelGroup[] = [];
    for (const [p, items] of Object.entries(itemsByProvider.value)) {
      if (items && items.length > 0) {
        groups.push({ label: PROVIDER_LABELS[p] ?? p, provider: p, items });
      }
    }
    // If the active model isn't in any group (e.g. set via .env.local), surface it
    // so the dropdown shows it rather than appearing blank.
    if (currentModel.value && !groups.some((g) => g.items.some((i) => i.value === currentModel.value))) {
      groups.push({
        label: "Current",
        provider: null,
        items: [{ label: currentModel.value, value: currentModel.value }],
      });
    }
    return groups;
  });

  // Filter groups by the selected provider.
  // Groups with provider: null (e.g. "Current") always appear.
  const filteredGroups = computed(() => {
    const p = provider.value;
    return modelGroups.value.filter((g) => g.provider === null || g.provider === p);
  });

  const suggestions = ref<ModelGroup[]>([]);

  function search(event: { query: string }) {
    const query = event.query.trim().toLowerCase();
    const source = filteredGroups.value;
    if (!query) {
      suggestions.value = [...source];
      return;
    }
    const filtered: ModelGroup[] = [];
    for (const group of source) {
      const matchingItems = group.items.filter(
        (i) => i.label.toLowerCase().includes(query) || i.value.toLowerCase().includes(query),
      );
      if (matchingItems && matchingItems.length > 0) {
        filtered.push({ label: group.label, provider: group.provider, items: matchingItems });
      }
    }
    suggestions.value = filtered;
  }

  const isOllamaModel = computed(() => currentModel.value.startsWith("ollama/"));

  // Single status for the Ollama model-picker footer, so the UI can render
  // loading / unreachable / reachable-but-empty instead of an empty dropdown.
  const ollamaStatus = computed<"idle" | "loading" | "unreachable" | "empty" | "ready">(() => {
    if (provider.value !== "ollama") return "idle";
    if (ollamaLoading.value) return "loading";
    if (ollamaSource.value === "unreachable") return "unreachable";
    if (ollamaSource.value === "live") {
      const items = itemsByProvider.value.ollama;
      return items && items.length > 0 ? "ready" : "empty";
    }
    return "idle";
  });

  return { modelGroups, filteredGroups, suggestions, search, isOllamaModel, ollamaStatus, reloadOllama };
}
