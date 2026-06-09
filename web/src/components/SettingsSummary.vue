<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import type { ThreadConfigModel } from "../apis";

interface WizardData {
  apiBaseUrl: string;
  model: ThreadConfigModel | null;
  vlm: {
    useMainModel: boolean;
    model: string | null;
    apiKey: string | null;
    baseUrl: string | null;
  };
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  autoScroll: boolean;
}

defineProps<{
  wizardData: WizardData;
}>();

const emit = defineEmits<{
  (e: "complete"): void;
  (e: "back"): void;
}>();

function maskKey(key: string | null): string {
  if (!key) return "—";
  if (key.length <= 8) return "••••••••";
  return key.slice(0, 4) + "••••" + key.slice(-4);
}
</script>

<template>
  <div class="flex flex-col gap-6 h-full">
    <p class="text-sm text-slate-500 dark:text-slate-400">Review your configuration before getting started.</p>

    <div class="space-y-3 text-sm">
      <!-- Connection -->
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">Server</span>
        <span class="font-mono">{{ wizardData.apiBaseUrl }}</span>
      </div>

      <!-- Model -->
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">Model</span>
        <span class="font-mono">{{ wizardData.model?.name ?? "—" }}</span>
      </div>
      <div
        v-if="wizardData.model?.api_key"
        class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2"
      >
        <span class="text-slate-500 dark:text-slate-400">API Key</span>
        <span class="font-mono">{{ maskKey(wizardData.model.api_key) }}</span>
      </div>
      <div
        v-if="wizardData.model?.base_url"
        class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2"
      >
        <span class="text-slate-500 dark:text-slate-400">Base URL</span>
        <span class="font-mono">{{ wizardData.model.base_url }}</span>
      </div>

      <!-- VLM -->
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">Vision</span>
        <span v-if="wizardData.vlm.useMainModel" class="font-mono">Same as model</span>
        <span v-else class="font-mono">{{ wizardData.vlm.model ?? "—" }}</span>
      </div>
      <div
        v-if="!wizardData.vlm.useMainModel && wizardData.vlm.apiKey"
        class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2"
      >
        <span class="text-slate-500 dark:text-slate-400">VLM API Key</span>
        <span class="font-mono">{{ maskKey(wizardData.vlm.apiKey) }}</span>
      </div>

      <!-- Advanced -->
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">Proxy</span>
        <span class="font-mono">{{ wizardData.proxy ?? "—" }}</span>
      </div>
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">Tavily</span>
        <span class="font-mono">{{ wizardData.tavilyApiKey ? maskKey(wizardData.tavilyApiKey) : "—" }}</span>
      </div>
      <div class="flex justify-between border-b border-gray-200 dark:border-gray-700 pb-2">
        <span class="text-slate-500 dark:text-slate-400">KB Autosave</span>
        <span class="font-mono">{{ wizardData.kbAutosave ? "On" : "Off" }}</span>
      </div>
      <div class="flex justify-between pb-2">
        <span class="text-slate-500 dark:text-slate-400">Auto-scroll</span>
        <span class="font-mono">{{ wizardData.autoScroll ? "On" : "Off" }}</span>
      </div>
    </div>

    <div class="flex pt-4 justify-between mt-auto">
      <Button label="Back" severity="secondary" @click="emit('back')" />
      <Button label="Start Using CopilotJ" @click="emit('complete')" />
    </div>
  </div>
</template>
