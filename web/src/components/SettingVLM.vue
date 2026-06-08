<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  useMainModel: boolean;
}>();

const emit = defineEmits<{
  (
    e: "update",
    value: { model: string | null; apiKey: string | null; baseUrl: string | null; useMainModel: boolean },
  ): void;
}>();

const useMainModel = ref(true);
const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");

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
      <ModelAutoComplete v-model="model" inputId="vlmModel" :disabled="useMainModel" />
    </FormItem>

    <FormItem v-if="!isOllamaModel" for="vlmApiKey" label="API Key">
      <InputText
        type="password"
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
