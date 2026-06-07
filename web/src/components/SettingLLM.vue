<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { ThreadConfigModel } from "../apis";
import { useModelGroups } from "../composables";

const props = defineProps<{
  model: ThreadConfigModel | null;
}>();

const emit = defineEmits<{
  (e: "update:model", value: ThreadConfigModel | null): void;
}>();

const useDefaultModel = ref(true);
const model = ref("");
const apiKey = ref("");
const baseUrl = ref("");

const { modelGroups, isOllamaModel } = useModelGroups(model);

// --- Sync from props ---
watch(
  props,
  (newProps) => {
    if (newProps.model) {
      model.value = newProps.model.name || "";
      apiKey.value = newProps.model.api_key || "";
      baseUrl.value = newProps.model.base_url || "";
      useDefaultModel.value = false;
    } else {
      model.value = "";
      apiKey.value = "";
      baseUrl.value = "";
      useDefaultModel.value = true;
    }
  },
  { immediate: true },
);

function submit() {
  if (useDefaultModel.value) {
    emit("update:model", null);
  } else {
    emit("update:model", {
      name: model.value,
      api_key: isOllamaModel.value ? null : apiKey.value || null,
      base_url: baseUrl.value || null,
    });
  }
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <FormItem for="defaultModel" label="Use Default Model" layout="row">
      <ToggleSwitch v-model="useDefaultModel" inputId="defaultModel" />
    </FormItem>

    <p v-if="useDefaultModel && props.model?.name" class="text-sm text-slate-500 dark:text-slate-400 -mt-4">
      Active model: <span class="font-mono">{{ props.model.name }}</span>
    </p>

    <FormItem for="model" label="Model">
      <Select
        v-model="model"
        inputId="model"
        :options="modelGroups"
        optionGroupLabel="label"
        optionGroupChildren="items"
        optionLabel="label"
        optionValue="value"
        :disabled="useDefaultModel"
        placeholder="Select a model"
        class="w-full"
      />
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
