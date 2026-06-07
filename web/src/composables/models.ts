/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { computed, onMounted, type Ref, ref } from "vue";

const staticGroups = [
  {
    label: "Anthropic",
    items: [
      { label: "Claude Opus 4.6", value: "claude-opus-4-6" },
      { label: "Claude Sonnet 4.6", value: "claude-sonnet-4-6" },
      { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    ],
  },
  {
    label: "OpenAI",
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

export function useModelGroups(currentModel: Ref<string>) {
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
    const groups = [...staticGroups];
    if (ollamaModels.value.length > 0) {
      groups.push({
        label: "Ollama (local)",
        items: ollamaModels.value.map((m) => ({ label: m, value: `ollama/${m}` })),
      });
    }
    // If the active model isn't in any group (e.g. set via .env.local), surface it
    // so the dropdown shows it rather than appearing blank.
    if (currentModel.value && !groups.some((g) => g.items.some((i) => i.value === currentModel.value))) {
      groups.push({ label: "Current", items: [{ label: currentModel.value, value: currentModel.value }] });
    }
    return groups;
  });

  const isOllamaModel = computed(() => currentModel.value.startsWith("ollama/"));

  return { modelGroups, isOllamaModel };
}
