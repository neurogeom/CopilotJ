<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import { useModelGroups } from "../composables";

const props = defineProps<{
  model: ThreadConfigModel | null;
  serverConfig: ServerConfig | null;
}>();

const emit = defineEmits<{
  (e: "update:model", value: ThreadConfigModel | null): void;
}>();

const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");

const { modelGroups, isOllamaModel } = useModelGroups(model);

const isValid = computed(() => !!model.value);

// Pre-fill from server config or existing model prop
watch(
  [() => props.serverConfig, () => props.model],
  ([cfg, existingModel]) => {
    if (cfg?.model) {
      model.value = cfg.model.name || "";
      baseUrl.value = cfg.model.base_url || "";
    } else if (existingModel) {
      model.value = existingModel.name || "";
      apiKey.value = existingModel.api_key || "";
      baseUrl.value = existingModel.base_url || "";
    }
  },
  { immediate: true },
);

function getModelValue(): ThreadConfigModel {
  return {
    name: model.value,
    api_key: isOllamaModel.value ? null : apiKey.value || null,
    base_url: baseUrl.value || null,
  };
}

defineExpose({ isValid, getModelValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Choose the primary language model for your conversations.</p>

    <FormItem for="model" label="Model" required>
      <Select
        v-model="model"
        inputId="model"
        :options="modelGroups"
        optionGroupLabel="label"
        optionGroupChildren="items"
        optionLabel="label"
        optionValue="value"
        placeholder="Select a model"
        class="w-full"
      />
    </FormItem>

    <FormItem v-if="!isOllamaModel" for="apiKey" label="API Key">
      <InputText type="password" v-model="apiKey" inputId="apiKey" placeholder="Enter your API key" class="w-full" />
    </FormItem>

    <FormItem for="baseUrl" label="Base URL">
      <InputText
        type="text"
        v-model="baseUrl"
        inputId="baseUrl"
        placeholder="https://api.example.com/v1 (optional)"
        class="w-full"
      />
    </FormItem>
  </div>
</template>
