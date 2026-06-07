<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref } from "vue";
import { useModelGroups } from "../composables";

const props = defineProps<{
  useMainModel: boolean;
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
}>();

const emit = defineEmits<{
  (
    e: "update",
    value: { useMainModel: boolean; model: string | null; apiKey: string | null; baseUrl: string | null },
  ): void;
}>();

const useMainModel = ref(props.useMainModel);
const model = ref(props.model || "");
const apiKey = ref(props.apiKey || "");
const baseUrl = ref(props.baseUrl || "");

const { modelGroups, isOllamaModel } = useModelGroups(model);

const isValid = computed(() => useMainModel.value || !!model.value);

function getVlmValue() {
  return {
    useMainModel: useMainModel.value,
    model: useMainModel.value ? null : model.value,
    apiKey: useMainModel.value ? null : isOllamaModel.value ? null : apiKey.value || null,
    baseUrl: useMainModel.value ? null : baseUrl.value || null,
  };
}

defineExpose({ isValid, getVlmValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Configure the vision model used for image analysis tasks.</p>

    <FormItem for="useMainVlm" label="Use main model for vision" layout="row">
      <ToggleSwitch v-model="useMainModel" inputId="useMainVlm" />
    </FormItem>

    <p v-if="useMainModel" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Vision tasks will use the same model and API key as the main model.
    </p>

    <template v-if="!useMainModel">
      <FormItem for="vlmModel" label="Model" required>
        <Select
          v-model="model"
          inputId="vlmModel"
          :options="modelGroups"
          optionGroupLabel="label"
          optionGroupChildren="items"
          optionLabel="label"
          optionValue="value"
          placeholder="Select a model"
          class="w-full"
        />
      </FormItem>

      <FormItem v-if="!isOllamaModel" for="vlmApiKey" label="API Key">
        <InputText
          type="password"
          v-model="apiKey"
          inputId="vlmApiKey"
          placeholder="Enter your API key"
          class="w-full"
        />
      </FormItem>

      <FormItem for="vlmBaseUrl" label="Base URL">
        <InputText
          type="text"
          v-model="baseUrl"
          inputId="vlmBaseUrl"
          placeholder="https://api.example.com/v1 (optional)"
          class="w-full"
        />
      </FormItem>
    </template>
  </div>
</template>
