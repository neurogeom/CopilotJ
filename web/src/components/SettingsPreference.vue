<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref, watch } from "vue";

interface PrefValue {
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  autoScroll: boolean;
}

// Two-way preferences value. Live-bound by the parent; edits emit on change.
const value = defineModel<PrefValue>({ required: true });

const proxy = ref("");
const tavilyApiKey = ref("");
const kbAutosave = ref(false);
const autoScroll = ref(true);

// Seed fields from the model value (initial + whenever the parent resets it).
watch(
  value,
  (v) => {
    if (!v) return;
    proxy.value = v.proxy || "";
    tavilyApiKey.value = v.tavilyApiKey || "";
    kbAutosave.value = v.kbAutosave;
    autoScroll.value = v.autoScroll;
  },
  { immediate: true },
);

// Emit the composed value live on every field change (guarded to avoid a
// seed↔emit loop).
watch([proxy, tavilyApiKey, kbAutosave, autoScroll], () => {
  const next = getValue();
  if (JSON.stringify(value.value) !== JSON.stringify(next)) {
    value.value = next;
  }
});

function getValue(): PrefValue {
  return {
    proxy: proxy.value || null,
    tavilyApiKey: tavilyApiKey.value || null,
    kbAutosave: kbAutosave.value,
    autoScroll: autoScroll.value,
  };
}

defineExpose({ getValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Optional settings. You can configure these later.</p>

    <FormItem for="proxy" label="HTTP Proxy">
      <InputText
        type="text"
        v-model="proxy"
        inputId="proxy"
        placeholder="http://127.0.0.1:8080 (optional)"
        class="w-full"
      />
    </FormItem>

    <FormItem for="tavilyApiKey" label="Tavily API Key">
      <InputText
        type="password"
        v-model="tavilyApiKey"
        inputId="tavilyApiKey"
        placeholder="tvly-xxxxxxxx (optional, for web search)"
        class="w-full"
      />
    </FormItem>

    <!-- "Auto-save to Knowledge Bank" toggle — hidden; value always sent as null. TODO: temporarily disabled -->
    <!-- <FormItem for="kbAutosave" label="Auto-save to Knowledge Bank" layout="row">
      <ToggleSwitch v-model="kbAutosave" inputId="kbAutosave" />
    </FormItem> -->

    <FormItem for="autoScroll" label="Auto-scroll to Bottom" layout="row">
      <ToggleSwitch v-model="autoScroll" inputId="autoScroll" />
    </FormItem>
  </div>
</template>
