<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-vue";
import { getModelCapabilities } from "../apis";
import {
  PROVIDER_OPTIONS,
  inferProvider,
  resolveBaseUrl,
  hasCustomBaseUrl,
  persistBaseUrl,
  getDefaultBaseUrl,
} from "../composables";

const props = defineProps<{
  useMainModel: boolean;
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  provider: string | null;
  mainModelName: string | null;
  showSubmitButton?: boolean;
}>();

const emit = defineEmits<{
  (
    e: "update",
    value: {
      useMainModel: boolean;
      model: string | null;
      apiKey: string | null;
      baseUrl: string | null;
      provider: string | null;
    },
  ): void;
}>();

const mainModelSupportsVision = ref<boolean | null>(null);
const checkingVision = ref(false);

const useMainModel = ref(props.useMainModel);
const model = ref(props.model || "");
const apiKey = ref(props.apiKey || "");
const provider = ref(props.provider || (props.model ? inferProvider(props.model) : ""));
// Show the effective endpoint (stored override or the provider default), but
// only reveal "Advanced settings" for a genuine override.
const baseUrl = ref(resolveBaseUrl(provider.value, props.baseUrl));
const showAdvanced = ref(hasCustomBaseUrl(provider.value, props.baseUrl));

// Template ref to the Ollama model picker so the Base URL field can trigger an
// immediate refresh on blur via the picker's exposed reloadOllama().
const ollamaModelRef = ref<{ reloadOllama: () => void } | null>(null);

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

const isValid = computed(() => useMainModel.value || !!model.value);

async function checkVisionSupport(modelName: string | null) {
  if (!modelName) {
    mainModelSupportsVision.value = null;
    checkingVision.value = false;
    return;
  }
  checkingVision.value = true;
  try {
    const caps = await getModelCapabilities(modelName);
    mainModelSupportsVision.value = caps.supports_vision;
    useMainModel.value = caps.supports_vision;
  } catch {
    mainModelSupportsVision.value = null;
  } finally {
    checkingVision.value = false;
  }
}

watch(() => props.mainModelName, checkVisionSupport, { immediate: true });

// Clear the model and pre-fill the provider's default base URL when the user
// switches provider — models and endpoints are provider-specific.
function onProviderChange() {
  model.value = "";
  baseUrl.value = getDefaultBaseUrl(provider.value);
}

function getVlmValue() {
  return {
    useMainModel: useMainModel.value,
    model: useMainModel.value ? null : model.value,
    apiKey: useMainModel.value ? null : isOllamaModel.value ? null : apiKey.value || null,
    baseUrl: useMainModel.value ? null : persistBaseUrl(provider.value, baseUrl.value),
    provider: useMainModel.value ? null : provider.value || inferProvider(model.value),
  };
}

function submit() {
  emit("update", getVlmValue());
}

defineExpose({ isValid, getVlmValue });
</script>

<template>
  <div class="flex flex-col gap-6 h-full w-full">
    <hr class="border-slate-200 dark:border-slate-700" />

    <p class="text-sm text-slate-500 dark:text-slate-400">Configure the vision model used for image analysis tasks.</p>

    <div class="flex items-center gap-3">
      <FormItem for="useMainVlm" label="Use main model for vision" layout="row" class="flex-1">
        <ToggleSwitch
          v-model="useMainModel"
          inputId="useMainVlm"
          :disabled="checkingVision || mainModelSupportsVision === false"
        />
      </FormItem>
      <ProgressSpinner v-if="checkingVision" style="width: 20px; height: 20px" strokeWidth="4" />
    </div>

    <p v-if="checkingVision" class="text-sm text-slate-400 -mt-4">Checking vision capability…</p>

    <p v-else-if="mainModelSupportsVision === false" class="text-sm text-amber-600 -mt-4">
      The selected model does not support image input. Please configure a separate vision model below.
    </p>

    <p v-else-if="useMainModel" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Vision tasks will use the same model and API key as the main model.
    </p>

    <template v-if="!useMainModel">
      <FormItem for="vlmProvider" label="Provider" required>
        <Select
          v-model="provider"
          :options="PROVIDER_OPTIONS"
          optionLabel="label"
          optionValue="value"
          inputId="vlmProvider"
          placeholder="Select a provider"
          @change="onProviderChange"
          class="w-full"
        />
      </FormItem>

      <!-- Ollama: Base URL first (needed to list models), then Model; no API key. -->
      <template v-if="provider === 'ollama'">
        <FormItem for="vlmBaseUrl" label="Base URL" required>
          <InputText
            type="text"
            v-model="baseUrl"
            inputId="vlmBaseUrl"
            placeholder="http://localhost:11434"
            @blur="ollamaModelRef?.reloadOllama()"
            class="w-full"
          />
        </FormItem>
        <FormItem for="vlmModel" label="Model" required>
          <ModelAutoComplete
            ref="ollamaModelRef"
            v-model="model"
            inputId="vlmModel"
            :provider="provider"
            :base-url="baseUrl"
          />
        </FormItem>
      </template>

      <!-- OpenAI-compatible: Base URL is required (right under Provider); model is free text. -->
      <template v-else-if="provider === 'openai-compatible'">
        <FormItem for="vlmBaseUrl" label="Base URL" required>
          <InputText
            type="text"
            v-model="baseUrl"
            inputId="vlmBaseUrl"
            placeholder="https://your-endpoint.com/v1"
            class="w-full"
          />
        </FormItem>
        <FormItem for="vlmModel" label="Model" required>
          <InputText type="text" v-model="model" inputId="vlmModel" placeholder="Enter the model name" class="w-full" />
        </FormItem>
        <FormItem for="vlmApiKey" label="API Key">
          <InputText
            type="password"
            v-model="apiKey"
            inputId="vlmApiKey"
            placeholder="Enter your API key"
            class="w-full"
          />
        </FormItem>
      </template>

      <!-- Cloud providers: Model, API Key, then Base URL (advanced, collapsed). -->
      <template v-else-if="provider">
        <FormItem for="vlmModel" label="Model" required>
          <ModelAutoComplete v-model="model" inputId="vlmModel" :provider="provider" />
        </FormItem>
        <FormItem for="vlmApiKey" label="API Key">
          <InputText
            type="password"
            v-model="apiKey"
            inputId="vlmApiKey"
            placeholder="Enter your API key"
            class="w-full"
          />
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
          <FormItem v-if="showAdvanced" for="vlmBaseUrl" label="Base URL" class="mt-2">
            <InputText
              type="text"
              v-model="baseUrl"
              inputId="vlmBaseUrl"
              placeholder="https://api.example.com/v1"
              class="w-full"
            />
          </FormItem>
        </div>
      </template>
    </template>

    <Button v-if="showSubmitButton" label="Submit" @click="submit" />
  </div>
</template>
