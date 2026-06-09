/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { computed, onMounted, type Ref, ref } from "vue";

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

const staticGroups: ModelGroup[] = [
  {
    label: "Anthropic",
    provider: "anthropic",
    items: [
      { label: "Claude Opus 4.6", value: "claude-opus-4-6" },
      { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6" },
      { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    ],
  },
  {
    label: "OpenAI",
    provider: "openai",
    items: [
      { label: "GPT-5.4", value: "gpt-5.4" },
      { label: "GPT-5.4 mini", value: "gpt-5.4-mini" },
      { label: "GPT-5", value: "gpt-5" },
      { label: "GPT-5 mini", value: "gpt-5-mini" },
      { label: "GPT-5 nano", value: "gpt-5-nano" },
      { label: "GPT-4.1", value: "gpt-4.1" },
      { label: "GPT-4.1 mini", value: "gpt-4.1-mini" },
      { label: "GPT-4.1 nano", value: "gpt-4.1-nano" },
      { label: "GPT-4o", value: "gpt-4o" },
      { label: "GPT-4o mini", value: "gpt-4o-mini" },
    ],
  },
  {
    label: "Google",
    provider: "gemini",
    items: [
      { label: "Gemini 3.1 Pro", value: "gemini-3.1-pro-preview" },
      { label: "Gemini 3 Flash", value: "gemini-3-flash-preview" },
      { label: "Gemini 3.1 Flash Lite", value: "gemini-3.1-flash-lite-preview" },
      { label: "Gemini 2.5 Pro", value: "gemini-2.5-pro" },
      { label: "Gemini 2.5 Flash", value: "gemini-2.5-flash" },
      { label: "Gemini 2.5 Flash Lite", value: "gemini-2.5-flash-lite" },
    ],
  },
];

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

export function useModelGroups(currentModel: Ref<string>, provider: Ref<string>) {
  const ollamaModels = ref<string[]>([]);

  onMounted(async () => {
    try {
      const resp = await fetch("http://localhost:11434/api/tags", {
        signal: AbortSignal.timeout(2000),
      });
      if (resp.ok) {
        const data = await resp.json();
        ollamaModels.value = (data.models ?? []).map((m: { name: string }) => m.name);
      }
    } catch {
      // Ollama not running or unavailable — silently ignore
    }
  });

  const modelGroups = computed(() => {
    const groups: ModelGroup[] = [...staticGroups];
    if (ollamaModels.value.length > 0) {
      groups.push({
        label: "Ollama (local)",
        provider: "ollama",
        items: ollamaModels.value.map((m) => ({ label: m, value: `ollama/${m}` })),
      });
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
