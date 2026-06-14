<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
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

const { suggestions, search, isOllamaModel } = useModelGroups(
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

defineExpose({ isOllamaModel });
</script>

<template>
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
</template>
