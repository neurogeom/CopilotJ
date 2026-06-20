<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-vue";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import {
  PROVIDER_OPTIONS,
  inferProvider,
  resolveBaseUrl,
  hasCustomBaseUrl,
  persistBaseUrl,
  getDefaultBaseUrl,
} from "../composables";

const props = withDefaults(
  defineProps<{
    model: ThreadConfigModel | null;
    /** Wizard: server config used to pre-fill the model. Settings: leave null. */
    serverConfig?: ServerConfig | null;
    /** Settings: name of the active (server) model, shown when "use default" is on. */
    serverModelName?: string | null;
    /** Settings: show the per-tab Submit button. Hidden in the wizard (uses Next). */
    showSubmitButton?: boolean;
  }>(),
  {
    serverConfig: null,
    serverModelName: null,
    showSubmitButton: false,
  },
);

const emit = defineEmits<{
  (e: "update:model", value: ThreadConfigModel | null): void;
}>();

const useDefaultModel = ref(false);
const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");
const provider = ref("");
const showAdvanced = ref(false);

// Template ref to the Ollama model picker so the Base URL field can trigger an
// immediate refresh on blur via the picker's exposed reloadOllama().
const ollamaModelRef = ref<{ reloadOllama: () => void } | null>(null);

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

// Valid when a default is in use or a model name is entered.
const isValid = computed(() => useDefaultModel.value || !!model.value);

// Pre-fill from server config (wizard) or the existing model prop (settings).
watch(
  [() => props.serverConfig, () => props.model],
  ([cfg, existingModel]) => {
    let stored = "";
    let name = "";
    if (cfg?.model) {
      name = cfg.model.name || "";
      provider.value = cfg.model.provider || inferProvider(name);
      apiKey.value = cfg.model.api_key || "";
      stored = cfg.model.base_url || "";
      useDefaultModel.value = false;
    } else if (existingModel) {
      name = existingModel.name || "";
      provider.value = existingModel.provider || inferProvider(name);
      apiKey.value = existingModel.api_key || "";
      stored = existingModel.base_url || "";
      useDefaultModel.value = existingModel.api_key == null;
    } else {
      useDefaultModel.value = true;
    }
    model.value = name;
    // Show the effective endpoint (stored override or the provider default),
    // but only reveal "Advanced settings" for a genuine override.
    baseUrl.value = resolveBaseUrl(provider.value, stored);
    showAdvanced.value = hasCustomBaseUrl(provider.value, stored);
  },
  { immediate: true },
);

// Clear the model and pre-fill the provider's default base URL when the user
// switches provider — models and endpoints are provider-specific.
function onProviderChange() {
  model.value = "";
  baseUrl.value = getDefaultBaseUrl(provider.value);
}

function getModelValue(): ThreadConfigModel | null {
  if (useDefaultModel.value) {
    return null;
  }
  return {
    name: model.value,
    api_key: isOllamaModel.value ? null : apiKey.value || null,
    base_url: persistBaseUrl(provider.value, baseUrl.value),
    provider: provider.value || inferProvider(model.value),
  };
}

function submit() {
  emit("update:model", getModelValue());
}

defineExpose({ isValid, getModelValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Choose the primary language model for your conversations.</p>

    <FormItem for="defaultModel" label="Use Default Model" layout="row">
      <ToggleSwitch v-model="useDefaultModel" inputId="defaultModel" />
    </FormItem>

    <p v-if="useDefaultModel && serverModelName" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Active model: <span class="font-mono">{{ serverModelName }}</span>
    </p>

    <FormItem for="provider" label="Provider" :required="!useDefaultModel">
      <Select
        v-model="provider"
        :options="PROVIDER_OPTIONS"
        optionLabel="label"
        optionValue="value"
        inputId="provider"
        placeholder="Select a provider"
        :disabled="useDefaultModel"
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
          :disabled="useDefaultModel"
          @blur="ollamaModelRef?.reloadOllama()"
          class="w-full"
        />
      </FormItem>
      <FormItem for="model" label="Model" :required="!useDefaultModel">
        <ModelAutoComplete
          ref="ollamaModelRef"
          v-model="model"
          inputId="model"
          :disabled="useDefaultModel"
          :provider="provider"
          :base-url="baseUrl"
        />
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
          :disabled="useDefaultModel"
          class="w-full"
        />
      </FormItem>
      <FormItem for="model" label="Model" :required="!useDefaultModel">
        <InputText
          type="text"
          v-model="model"
          inputId="model"
          placeholder="Enter the model name"
          :disabled="useDefaultModel"
          class="w-full"
        />
      </FormItem>
      <FormItem for="apiKey" label="API Key">
        <InputText
          type="password"
          v-model="apiKey"
          inputId="apiKey"
          placeholder="Enter your API key"
          :disabled="useDefaultModel"
          class="w-full"
        />
      </FormItem>
    </template>

    <!-- Cloud providers: Model, API Key, then Base URL (advanced, collapsed). -->
    <template v-else-if="provider">
      <FormItem for="model" label="Model" :required="!useDefaultModel">
        <ModelAutoComplete v-model="model" inputId="model" :disabled="useDefaultModel" :provider="provider" />
      </FormItem>
      <FormItem for="apiKey" label="API Key">
        <InputText
          type="password"
          v-model="apiKey"
          inputId="apiKey"
          placeholder="Enter your API key"
          :disabled="useDefaultModel"
          class="w-full"
        />
      </FormItem>
      <div>
        <button
          type="button"
          :disabled="useDefaultModel"
          class="flex select-none items-center gap-1 text-sm transition-colors"
          :class="
            useDefaultModel
              ? 'cursor-not-allowed text-slate-300 dark:text-slate-600'
              : 'cursor-pointer text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
          "
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
            :disabled="useDefaultModel"
            class="w-full"
          />
        </FormItem>
      </div>
    </template>

    <Button v-if="showSubmitButton" label="Submit" :disabled="!isValid" @click="submit" />
  </div>
</template>
