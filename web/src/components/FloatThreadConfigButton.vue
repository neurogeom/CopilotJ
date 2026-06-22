<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconSettings } from "@tabler/icons-vue";
import { ref } from "vue";
import type { ThreadConfig, ThreadConfigModel } from "../apis";
import { useThread } from "../store";
import FloatIconButton from "./FloatIconButton.vue";
import SettingsModel from "./SettingsModel.vue";

const props = defineProps<{
  thread: ReturnType<typeof useThread>;
}>();

const showConfig = ref(false);
const config = ref<ThreadConfig | null>(null);

async function onClick() {
  if (showConfig.value) {
    showConfig.value = false;
    return;
  }

  if (props.thread.config) {
    config.value = props.thread.config;
    showConfig.value = true;
  }
}

async function submitModel(newModel: ThreadConfigModel) {
  await props.thread.updateConfig({ model: newModel });
  showConfig.value = false;
}
</script>

<template>
  <FloatIconButton :icon="IconSettings" :disabled="thread.loading" @click="onClick" />

  <Dialog v-model:visible="showConfig" modal header="Thread Configuration">
    <SettingsModel
      v-if="config"
      :model="config?.model"
      :server-model-name="null"
      :show-submit-button="true"
      @update:model="submitModel"
    />
  </Dialog>
</template>
