<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { IconChevronDown, IconChevronRight } from "@tabler/icons-vue";
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
const provider = ref("");
const showAdvanced = ref(false);

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
      provider.value = "";
    }
    showAdvanced.value = !!baseUrl.value;
  },
  { immediate: true },
);

// Clear the model when the user switches provider — models are provider-specific.
function onProviderChange() {
  model.value = "";
}

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
      provider: provider.value || inferProvider(model.value),
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
          class="w-full"
        />
      </FormItem>
      <FormItem for="model" label="Model">
        <ModelAutoComplete
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
      <FormItem for="model" label="Model">
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
      <FormItem for="model" label="Model">
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

    <Button label="Submit" @click="submit" />
  </div>
</template>
