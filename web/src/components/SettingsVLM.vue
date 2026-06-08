<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref } from "vue";

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

const agreed = ref(false);
const showConfig = ref(false);

const useMainModel = ref(props.useMainModel);
const model = ref(props.model || "");
const apiKey = ref(props.apiKey || "");
const baseUrl = ref(props.baseUrl || "");

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

const isValid = computed(() => useMainModel.value || !!model.value);

function showPrivacy() {
  showConfig.value = false;
}

function showVisionConfig() {
  showConfig.value = true;
}

function next() {
  emit("update", {
    useMainModel: useMainModel.value,
    model: useMainModel.value ? null : model.value,
    apiKey: useMainModel.value ? null : isOllamaModel.value ? null : apiKey.value || null,
    baseUrl: useMainModel.value ? null : baseUrl.value || null,
  });
}

function back() {
  if (showConfig.value) {
    showConfig.value = false;
  } else {
    emit("back");
  }
}
</script>

<template>
  <!-- Page 1: Privacy consent -->
  <div v-if="!showConfig" class="flex flex-col gap-6 h-full">
    <div class="flex items-center gap-3 text-amber-500">
      <i class="pi pi-exclamation-triangle text-2xl" />
      <h3 class="text-lg font-semibold">Privacy Notice</h3>
    </div>

    <div class="text-sm text-slate-600 dark:text-slate-300 leading-relaxed space-y-3">
      <p>
        Enabling the Vision feature allows CopilotJ to capture screenshots from ImageJ and
        <strong>send them to an external AI model</strong> for visual analysis. Please be aware of the following:
      </p>
      <ul class="list-disc list-inside space-y-1 pl-2">
        <li>
          <strong>Image data is transmitted to a third-party AI provider</strong> (e.g. OpenAI, Anthropic, Google) over
          the internet. The confidentiality of your data depends on that provider's data handling and retention
          policies.
        </li>
        <li>
          Screenshots may contain
          <strong>file paths, window titles, patient identifiers, or other sensitive metadata</strong> visible in the
          ImageJ interface, in addition to the image content itself.
        </li>
        <li>
          CopilotJ has <strong>no control over how the provider stores, logs, or uses the transmitted data</strong>. If
          you work with confidential, proprietary, or personally identifiable data, consult your organization's data
          governance policy before enabling this feature.
        </li>
      </ul>
      <p class="text-slate-500 dark:text-slate-400">
        You can disable Vision at any time by removing the configuration from your settings.
      </p>
    </div>

    <div class="flex items-center gap-2 mt-2">
      <Checkbox v-model="agreed" binary inputId="visionAgree" />
      <label for="visionAgree" class="text-sm cursor-pointer select-none">
        I understand the privacy implications and wish to enable Vision features.
      </label>
    </div>

    <div class="flex pt-4 justify-between mt-auto">
      <Button label="Back" severity="secondary" @click="back" />
      <Button label="Next" :disabled="!agreed" @click="showVisionConfig" />
    </div>
  </div>

  <!-- Page 2: Vision model configuration -->
  <div v-else class="flex flex-col gap-6 h-full">
    <div class="flex items-center gap-2">
      <Button icon="pi pi-arrow-left" text rounded severity="secondary" @click="showPrivacy" />
      <p class="text-sm text-slate-500 dark:text-slate-400">
        Configure the vision model used for image analysis tasks.
      </p>
    </div>

    <FormItem for="useMainVlm" label="Use main model for vision" layout="row">
      <ToggleSwitch v-model="useMainModel" inputId="useMainVlm" />
    </FormItem>

    <p v-if="useMainModel" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Vision tasks will use the same model and API key as the main model.
    </p>

    <template v-if="!useMainModel">
      <FormItem for="vlmModel" label="Model" required>
        <ModelAutoComplete v-model="model" inputId="vlmModel" />
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

    <div class="flex pt-4 justify-between mt-auto">
      <Button label="Back" severity="secondary" @click="showPrivacy" />
      <Button label="Next" :disabled="!isValid" @click="next" />
    </div>
  </div>
</template>
