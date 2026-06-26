<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import Chatbox from "../components/Chatbox.vue";
import Settings from "../components/Settings.vue";
import Wizard from "../components/Wizard.vue";
import Sidebar from "../components/Sidebar.vue";
import { useConfig, useSettings, useSystemState } from "../store";

const settings = useSettings();
const state = useSystemState();
const config = useConfig();

const isFirstRun = computed(() => !localStorage.getItem("copilotj_config"));

onMounted(async () => {
  // First-time users: open the setup wizard
  if (isFirstRun.value) {
    state.wizardMode = true;
    return;
  }

  state.testBackendConnection();

  // 1. If config store has a user-configured model, apply it
  if (settings.model === null && config.data.defaultModel) {
    settings.setModel(config.data.defaultModel);
  }

  // 2. Fall back to the server's env config for all settings (model/VLM/vision
  // availability, proxy). Failures are swallowed here, same as before: the
  // connection warning will show and resolve on retry.
  await config.applyServerConfig();
});

const chatbox = ref<InstanceType<typeof Chatbox> | null>(null);
const settingsRef = ref<InstanceType<typeof Settings> | null>(null);

function onSetupComplete() {
  state.wizardMode = false;
  state.testBackendConnection();
}

function startNewThread() {
  chatbox.value?.reset();
}

function clickPost(postId: string) {
  chatbox.value?.scrollToPost(postId);
}
</script>

<template>
  <div class="flex h-screen w-full bg-gray-100 text-slate-950 dark:text-slate-100 dark:bg-black">
    <!-- Sidebar -->
    <Sidebar @startNewThread="startNewThread" @click-post="clickPost" />

    <!-- Chatbox -->
    <Chatbox ref="chatbox" :expandSidebar="settings.expandSidebar" @toggleSidebar="settings.toggleAutoScroll" />

    <!-- Settings Dialog (Tabs) -->
    <Dialog
      v-model:visible="state.showSettings"
      modal
      header="Settings"
      class="copilotj-settings-dialog w-[960px] max-w-[95vw] h-[780px] max-h-[90vh]"
    >
      <Settings ref="settingsRef" />
    </Dialog>

    <!-- Setup Wizard Dialog (Stepper) -->
    <Dialog
      v-model:visible="state.wizardMode"
      modal
      header="Welcome to CopilotJ"
      :closable="false"
      class="copilotj-settings-dialog w-[960px] max-w-[95vw] h-[780px] max-h-[90vh]"
    >
      <Wizard @complete="onSetupComplete" />
    </Dialog>

    <ConfirmPopup />
  </div>
</template>
