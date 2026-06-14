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
  { label: "OpenAI-compatible", value: "openai-compatible" },
];

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
    let items: { label: string; value: string }[] = [];
    try {
      const data = await listProviderModels("ollama", ollamaBaseUrl?.value);
      items = data.source === "live" ? toItems("ollama", data.models) : [];
    } catch {
      items = [];
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

  return { modelGroups, filteredGroups, suggestions, search, isOllamaModel };
}
