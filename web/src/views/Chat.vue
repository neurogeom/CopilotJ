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
import { getServerConfig, isUseServer } from "../apis";
import type { ThreadConfigModel } from "../apis";
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

  // 2. Fall back to server's env config for all settings
  try {
    const serverConfig = await getServerConfig();

    // Store server model/VLM for runtime use (not persisted to localStorage)
    config.setServerModel(serverConfig.model);
    config.setServerVlm(serverConfig.vlm);

    // Model — if nothing is configured yet but the server has a model, default
    // to "use the server's model" (an explicit, persisted choice).
    if (settings.model === null && serverConfig.model !== null) {
      const useServer: ThreadConfigModel = { use_server: true };
      settings.setModel(useServer);
      config.setDefaultModel(useServer);
    }

    // Reconcile stale "use server" choices: if the loaded server has no model for
    // a slot, a persisted {use_server:true} / useServerVlm choice is no longer
    // valid — clear it so the UI asks for an explicit model.
    if (isUseServer(config.data.defaultModel) && !serverConfig.model) {
      config.setDefaultModel(null);
      settings.setModel(null);
    }
    if (config.data.vlm.useServerVlm && !serverConfig.vlm) {
      config.setVlm({ ...config.data.vlm, useServerVlm: false });
    }

    // Proxy
    if (config.data.proxy === null && serverConfig.proxy !== null) {
      config.setProxy(serverConfig.proxy);
    }

    // KB autosave
    // "Auto Save" disabled; value always sent as null. TODO: temporarily disabled
    // if (!config.data.kbAutosave && serverConfig.kb_autosave) {
    //   config.setKbAutosave(serverConfig.kb_autosave);
    // }

    // Vision: store the server's default for display only. The user's explicit
    // choice (visionEnabled) is authoritative; null defers to this server value.
    config.setServerVisionEnabled(serverConfig.vision_enabled);
  } catch {
    // Server may not be reachable yet; warning will show and resolve on retry.
  }
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
