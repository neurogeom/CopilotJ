<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref } from "vue";

const props = withDefaults(
  defineProps<{
    proxy: string | null;
    tavilyApiKey: string | null;
    kbAutosave: boolean;
    autoScroll: boolean;
    /** Settings: show the per-tab Save button. Hidden in the wizard (uses Next). */
    showSubmitButton?: boolean;
  }>(),
  { showSubmitButton: false },
);

const emit = defineEmits<{
  (
    e: "update",
    value: { proxy: string | null; tavilyApiKey: string | null; kbAutosave: boolean; autoScroll: boolean },
  ): void;
}>();

const proxy = ref(props.proxy || "");
const tavilyApiKey = ref(props.tavilyApiKey || "");
const kbAutosave = ref(props.kbAutosave);
const autoScroll = ref(props.autoScroll);

function getValue() {
  return {
    proxy: proxy.value || null,
    tavilyApiKey: tavilyApiKey.value || null,
    kbAutosave: kbAutosave.value,
    autoScroll: autoScroll.value,
  };
}

function submit() {
  emit("update", getValue());
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

    <Button v-if="showSubmitButton" label="Save" @click="submit" />
  </div>
</template>
