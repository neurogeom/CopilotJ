<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { getModelCapabilities } from "../apis";

const props = defineProps<{
  useMainModel: boolean;
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  mainModelName: string | null;
  visionEnabled?: boolean;
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
      visionEnabled: boolean;
    },
  ): void;
}>();

const agreed = ref(props.visionEnabled ?? false);
const mainModelSupportsVision = ref<boolean | null>(null);
const checkingVision = ref(false);

const useMainModel = ref(props.useMainModel);
const model = ref(props.model || "");
const apiKey = ref(props.apiKey || "");
const baseUrl = ref(props.baseUrl || "");

const isOllamaModel = computed(() => model.value.startsWith("ollama/"));

const isValid = computed(() => !agreed.value || useMainModel.value || !!model.value);

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

function getVlmValue() {
  return {
    useMainModel: useMainModel.value,
    model: useMainModel.value ? null : model.value,
    apiKey: useMainModel.value ? null : isOllamaModel.value ? null : apiKey.value || null,
    baseUrl: useMainModel.value ? null : baseUrl.value || null,
    visionEnabled: agreed.value,
  };
}

function submit() {
  emit("update", getVlmValue());
}

defineExpose({ isValid, getVlmValue });
</script>

<template>
  <div class="flex flex-col gap-6 h-full max-w-2xl">
    <!-- Privacy notice -->
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
        You can disable Vision at any time by unchecking the consent below.
      </p>
    </div>

    <div class="flex items-center gap-2 mt-2">
      <Checkbox v-model="agreed" binary inputId="visionAgree" />
      <label for="visionAgree" class="text-sm cursor-pointer select-none">
        I understand the privacy implications and wish to enable Vision features.
      </label>
    </div>

    <!-- Vision model configuration (shown after consent) -->
    <template v-if="agreed">
      <hr class="border-slate-200 dark:border-slate-700" />

      <p class="text-sm text-slate-500 dark:text-slate-400">
        Configure the vision model used for image analysis tasks.
      </p>

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
    </template>

    <Button v-if="showSubmitButton" label="Submit" @click="submit" />
  </div>
</template>
