<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconAlertTriangle, IconLoader2 } from "@tabler/icons-vue";
import { useModelGroups } from "../composables";

const props = withDefaults(
  defineProps<{
    modelValue: string;
    provider: string;
    disabled?: boolean;
    inputId?: string;
    placeholder?: string;
    /** Ollama host; only used when provider is "ollama" to list installed models. */
    baseUrl?: string;
  }>(),
  {
    disabled: false,
    inputId: undefined,
    placeholder: "Select or type a model",
    baseUrl: undefined,
  },
);

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();

// AutoComplete's v-model is polymorphic: string on free text, object on selection.
// We normalise it to always emit a plain string.
const internalValue = ref<string | { label: string; value: string }>(props.modelValue);

// Use a "committed" snapshot for the composable so the "Current" fallback group
// stays stable during active typing and only updates on blur / selection.
const committedModel = ref(props.modelValue);

const providerRef = computed(() => props.provider);
const baseUrlRef = computed(() => props.baseUrl);

const { suggestions, search, isOllamaModel, ollamaStatus, reloadOllama } = useModelGroups(
  computed(() => {
    const v = internalValue.value;
    return typeof v === "string" ? v : (v?.value ?? "");
  }),
  providerRef,
  baseUrlRef,
);

// Sync incoming prop changes.
watch(
  () => props.modelValue,
  (newVal) => {
    internalValue.value = newVal;
    committedModel.value = newVal;
  },
);

// Normalise outgoing value to always be a string.
watch(internalValue, (newVal) => {
  const normalized = typeof newVal === "string" ? newVal : (newVal?.value ?? "");
  if (normalized !== props.modelValue) {
    emit("update:modelValue", normalized);
  }
});

function onBlur() {
  const v = internalValue.value;
  committedModel.value = typeof v === "string" ? v : (v?.value ?? "");
}

defineExpose({ isOllamaModel, reloadOllama });
</script>

<template>
  <div>
    <AutoComplete
      v-model="internalValue"
      :suggestions="suggestions"
      @complete="search"
      @blur="onBlur"
      optionLabel="label"
      optionGroupLabel="label"
      optionGroupChildren="items"
      dropdown
      dropdownMode="blank"
      :forceSelection="false"
      :disabled="disabled"
      :inputId="inputId"
      :placeholder="placeholder"
      class="w-full"
    >
      <template #optiongroup="slotProps">
        <div class="font-bold text-sm text-slate-500 dark:text-slate-400 px-2 py-1">
          {{ slotProps.option.label }}
        </div>
      </template>
      <template #option="slotProps">
        <div class="flex items-center justify-between gap-2">
          <span>{{ slotProps.option.label }}</span>
          <span class="text-xs text-slate-400 font-mono">{{ slotProps.option.value }}</span>
        </div>
      </template>
    </AutoComplete>

    <!-- Ollama fetch status: surface failures instead of an empty dropdown. -->
    <p
      v-if="provider === 'ollama' && ollamaStatus !== 'idle' && ollamaStatus !== 'ready'"
      class="mt-1 flex items-center gap-1.5 text-xs"
      :class="
        ollamaStatus === 'unreachable' ? 'text-amber-600 dark:text-amber-500' : 'text-slate-500 dark:text-slate-400'
      "
    >
      <template v-if="ollamaStatus === 'loading'">
        <IconLoader2 size="14" class="animate-spin" />
        <span>Loading models from Ollama…</span>
      </template>
      <template v-else-if="ollamaStatus === 'unreachable'">
        <IconAlertTriangle size="14" />
        <span
          >Couldn't reach Ollama at <code class="font-mono">{{ baseUrl || "the configured host" }}</code
          >. Is it running?</span
        >
      </template>
      <template v-else>
        <span
          >Ollama is reachable but has no models installed. Run
          <code class="font-mono">ollama pull &lt;model&gt;</code>.</span
        >
      </template>
    </p>
  </div>
</template>
