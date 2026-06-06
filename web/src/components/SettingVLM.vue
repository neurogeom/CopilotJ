<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";

const props = defineProps<{
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  useMainModel: boolean;
}>();

const emit = defineEmits<{
  (e: "update", value: { model: string | null; apiKey: string | null; baseUrl: string | null; useMainModel: boolean }): void;
}>();

const useMainModel = ref(true);
const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");

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
    // Ollama not running
  }
});

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

const modelGroups = computed(() => {
  const groups = [...staticGroups];
  if (ollamaModels.value.length > 0) {
    groups.push({
      label: "Ollama (local)",
      items: ollamaModels.value.map((m) => ({ label: m, value: `ollama/${m}` })),
    });
  }
  if (model.value && !groups.some((g) => g.items.some((i) => i.value === model.value))) {
    groups.push({ label: "Current", items: [{ label: model.value, value: model.value }] });
  }
  return groups;
});

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

watch(
  props,
  (newProps) => {
    useMainModel.value = newProps.useMainModel;
    model.value = newProps.model || "";
    apiKey.value = newProps.apiKey || "";
    baseUrl.value = newProps.baseUrl || "";
  },
  { immediate: true },
);

function submit() {
  emit("update", {
    model: useMainModel.value ? null : model.value,
    apiKey: useMainModel.value ? null : isOllamaModel.value ? null : apiKey.value || null,
    baseUrl: useMainModel.value ? null : baseUrl.value || null,
    useMainModel: useMainModel.value,
  });
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <FormItem for="useMainVlm" label="Use main model for vision" layout="row">
      <ToggleSwitch v-model="useMainModel" inputId="useMainVlm" />
    </FormItem>

    <p v-if="useMainModel" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Vision tasks will use the same model and API key as the main model.
    </p>

    <FormItem for="vlmModel" label="Model">
      <Select
        v-model="model"
        inputId="vlmModel"
        :options="modelGroups"
        optionGroupLabel="label"
        optionGroupChildren="items"
        optionLabel="label"
        optionValue="value"
        :disabled="useMainModel"
        placeholder="Select a model"
        class="w-full"
      />
    </FormItem>

    <FormItem v-if="!isOllamaModel" for="vlmApiKey" label="API Key">
      <InputText
        type="text"
        v-model="apiKey"
        inputId="vlmApiKey"
        placeholder="Enter your API key"
        :disabled="useMainModel"
        class="w-full"
      />
    </FormItem>

    <FormItem for="vlmBaseUrl" label="Base URL">
      <InputText
        type="text"
        v-model="baseUrl"
        inputId="vlmBaseUrl"
        placeholder="https://api.example.com/v1 (optional)"
        :disabled="useMainModel"
        class="w-full"
      />
    </FormItem>

    <Button label="Submit" @click="submit" />
  </div>
</template>
