<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-vue";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import { PROVIDER_OPTIONS, inferProvider } from "../composables";

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
const provider = ref("");
const showAdvanced = ref(false);

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

const isValid = computed(() => !!model.value);

// Pre-fill from server config or existing model prop
watch(
  [() => props.serverConfig, () => props.model],
  ([cfg, existingModel]) => {
    if (cfg?.model) {
      model.value = cfg.model.name || "";
      baseUrl.value = cfg.model.base_url || "";
      provider.value = cfg.model.provider || inferProvider(cfg.model.name || "");
    } else if (existingModel) {
      model.value = existingModel.name || "";
      apiKey.value = existingModel.api_key || "";
      baseUrl.value = existingModel.base_url || "";
      provider.value = existingModel.provider || inferProvider(existingModel.name || "");
    }
    // Reveal Advanced when an existing base URL is present.
    showAdvanced.value = !!baseUrl.value;
  },
  { immediate: true },
);

// Clear the model when the user switches provider — models are provider-specific.
function onProviderChange() {
  model.value = "";
}

function getModelValue(): ThreadConfigModel {
  return {
    name: model.value,
    api_key: isOllamaModel.value ? null : apiKey.value || null,
    base_url: baseUrl.value || null,
    provider: provider.value || inferProvider(model.value),
  };
}

defineExpose({ isValid, getModelValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Choose the primary language model for your conversations.</p>

    <FormItem for="provider" label="Provider" required>
      <Select
        v-model="provider"
        :options="PROVIDER_OPTIONS"
        optionLabel="label"
        optionValue="value"
        inputId="provider"
        placeholder="Select a provider"
        @change="onProviderChange"
        class="w-full"
      />
    </FormItem>

    <!-- Ollama: Base URL first (needed to list models), then Model; no API key. -->
    <template v-if="provider === 'ollama'">
      <FormItem for="baseUrl" label="Base URL" required>
        <InputText
          type="text"
          v-model="baseUrl"
          inputId="baseUrl"
          placeholder="http://localhost:11434"
          class="w-full"
        />
      </FormItem>
      <FormItem for="model" label="Model" required>
        <ModelAutoComplete v-model="model" inputId="model" :provider="provider" :base-url="baseUrl" />
      </FormItem>
    </template>

    <!-- OpenAI-compatible: Base URL is required (right under Provider); model is free text. -->
    <template v-else-if="provider === 'openai-compatible'">
      <FormItem for="baseUrl" label="Base URL" required>
        <InputText
          type="text"
          v-model="baseUrl"
          inputId="baseUrl"
          placeholder="https://your-endpoint.com/v1"
          class="w-full"
        />
      </FormItem>
      <FormItem for="model" label="Model" required>
        <InputText type="text" v-model="model" inputId="model" placeholder="Enter the model name" class="w-full" />
      </FormItem>
      <FormItem for="apiKey" label="API Key">
        <InputText type="password" v-model="apiKey" inputId="apiKey" placeholder="Enter your API key" class="w-full" />
      </FormItem>
    </template>

    <!-- Cloud providers: Model, API Key, then Base URL (advanced, collapsed). -->
    <template v-else-if="provider">
      <FormItem for="model" label="Model" required>
        <ModelAutoComplete v-model="model" inputId="model" :provider="provider" />
      </FormItem>
      <FormItem for="apiKey" label="API Key">
        <InputText type="password" v-model="apiKey" inputId="apiKey" placeholder="Enter your API key" class="w-full" />
      </FormItem>
      <div>
        <button
          type="button"
          class="flex cursor-pointer select-none items-center gap-1 text-sm text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
          @click="showAdvanced = !showAdvanced"
        >
          Advanced settings
          <component :is="showAdvanced ? IconChevronDown : IconChevronRight" size="16" />
        </button>
        <FormItem v-if="showAdvanced" for="baseUrl" label="Base URL" class="mt-2">
          <InputText
            type="text"
            v-model="baseUrl"
            inputId="baseUrl"
            placeholder="https://api.example.com/v1"
            class="w-full"
          />
        </FormItem>
      </div>
    </template>
  </div>
</template>
