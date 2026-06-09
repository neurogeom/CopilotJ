<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { ThreadConfigModel } from "../apis";
import { PROVIDER_OPTIONS, inferProvider } from "../composables";

const props = defineProps<{
  model: ThreadConfigModel | null;
  serverModelName: string | null;
}>();

const emit = defineEmits<{
  (e: "update:model", value: ThreadConfigModel | null): void;
}>();

const useDefaultModel = ref(props.model?.api_key == null);
const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");
const provider = ref(PROVIDER_OPTIONS[0].value);

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

// --- Sync form fields from props (but not the useDefaultModel toggle) ---
watch(
  () => props.model,
  (newModel) => {
    if (newModel) {
      model.value = newModel.name || "";
      apiKey.value = newModel.api_key || "";
      baseUrl.value = newModel.base_url || "";
      provider.value = newModel.provider || inferProvider(newModel.name || "");
    } else {
      model.value = "";
      apiKey.value = "";
      baseUrl.value = "";
      provider.value = PROVIDER_OPTIONS[0].value;
    }
  },
  { immediate: true },
);

function submit() {
  if (useDefaultModel.value) {
    // Emit null so the parent clears the persisted defaultModel.
    // The parent sets settings.model to the server model instead.
    emit("update:model", null);
  } else {
    emit("update:model", {
      name: model.value,
      api_key: isOllamaModel.value ? null : apiKey.value || null,
      base_url: baseUrl.value || null,
      provider: provider.value,
    });
  }
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <FormItem for="defaultModel" label="Use Default Model" layout="row">
      <ToggleSwitch v-model="useDefaultModel" inputId="defaultModel" />
    </FormItem>

    <p v-if="useDefaultModel && serverModelName" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Active model: <span class="font-mono">{{ serverModelName }}</span>
    </p>

    <FormItem for="provider" label="Provider">
      <Select
        v-model="provider"
        :options="PROVIDER_OPTIONS"
        optionLabel="label"
        optionValue="value"
        inputId="provider"
        placeholder="Select a provider"
        :disabled="useDefaultModel"
        class="w-full"
      />
    </FormItem>

    <FormItem for="model" label="Model">
      <ModelAutoComplete v-model="model" inputId="model" :disabled="useDefaultModel" :provider="provider" />
    </FormItem>

    <FormItem v-if="!isOllamaModel" for="apiKey" label="API Key">
      <InputText
        type="password"
        v-model="apiKey"
        inputId="apiKey"
        placeholder="Enter your API key"
        :disabled="useDefaultModel"
        class="w-full"
      />
    </FormItem>

    <FormItem for="baseUrl" label="Base URL">
      <InputText
        type="text"
        v-model="baseUrl"
        inputId="baseUrl"
        placeholder="https://api.example.com/v1 (optional)"
        :disabled="useDefaultModel"
        class="w-full"
      />
    </FormItem>

    <Button label="Submit" @click="submit" />
  </div>
</template>
