<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-vue";
import { isExplicit } from "../apis";
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
    /** Wizard: server config used to pre-fill the model. Settings: leave null. */
    serverConfig?: ServerConfig | null;
    /** Settings: name of the active (server) model, shown when "use default" is on. */
    serverModelName?: string | null;
  }>(),
  {
    serverConfig: null,
    serverModelName: null,
  },
);

// Two-way model value: {use_server:true} or an explicit model. Live-bound by
// the parent (Settings draft / Wizard wizard.model); edits emit on every change.
const modelValue = defineModel<ThreadConfigModel | null>({ default: null });

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

// Model name surfaced when "Use server model" is on: the settings prop, or
// the wizard's server-config model.
const activeModelName = computed(() => props.serverModelName ?? props.serverConfig?.model?.name ?? null);

// Whether the server actually exposes a model. "Use server model" is only
// selectable (and only valid) when this is true.
const serverModelAvailable = computed(() => !!(props.serverModelName ?? props.serverConfig?.model?.name ?? null));

// Pre-fill only from the model value. The server's default is never written
// into the fields — "Use Default Model" (on) represents using it.
watch(
  [() => props.serverConfig, modelValue],
  ([, existingModel]) => {
    let stored = "";
    let name = "";
    if (existingModel && isExplicit(existingModel)) {
      name = existingModel.name || "";
      provider.value = existingModel.provider || inferProvider(name);
      apiKey.value = existingModel.api_key || "";
      stored = existingModel.base_url || "";
      useDefaultModel.value = false;
    } else {
      // null or {use_server:true} → use the server's model, but only when the
      // server actually has one; otherwise fall through to explicit (empty).
      useDefaultModel.value = serverModelAvailable.value;
    }
    model.value = name;
    // Show the effective endpoint (stored override or the provider default),
    // but only reveal "Advanced settings" for a genuine override.
    baseUrl.value = resolveBaseUrl(provider.value, stored);
    showAdvanced.value = hasCustomBaseUrl(provider.value, stored);
  },
  { immediate: true },
);

// If the server loses its model (e.g. the Wizard connects to a modelless
// server), drop a stale "Use server model" choice so the user must pick one.
watch(serverModelAvailable, (a) => {
  if (!a && useDefaultModel.value) useDefaultModel.value = false;
});

// Emit the composed value live whenever a field changes. The equality guard
// breaks the seed↔emit feedback loop (seeding from the parent re-derives the
// same fields, so it doesn't re-emit).
watch([useDefaultModel, model, apiKey, baseUrl, provider], () => {
  const next = getModelValue();
  if (JSON.stringify(modelValue.value) !== JSON.stringify(next)) {
    modelValue.value = next;
  }
});

// Clear the model and pre-fill the provider's default base URL when the user
// switches provider — models and endpoints are provider-specific.
function onProviderChange() {
  model.value = "";
  baseUrl.value = getDefaultBaseUrl(provider.value);
}

function getModelValue(): ThreadConfigModel {
  if (useDefaultModel.value) {
    return { use_server: true };
  }
  return {
    name: model.value,
    api_key: isOllamaModel.value ? null : apiKey.value || null,
    base_url: persistBaseUrl(provider.value, baseUrl.value),
    provider: provider.value || inferProvider(model.value),
  };
}

defineExpose({ getModelValue });
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Choose the primary language model for your conversations.</p>

    <FormItem for="defaultModel" label="Use server model" layout="row">
      <ToggleSwitch v-model="useDefaultModel" inputId="defaultModel" :disabled="!serverModelAvailable" />
    </FormItem>

    <p v-if="useDefaultModel && activeModelName" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Active model: <span class="font-mono">{{ activeModelName }}</span>
    </p>
    <p v-else-if="!serverModelAvailable" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      No model configured on the server — choose one below.
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
  </div>
</template>
