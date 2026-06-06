<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { onMounted, ref } from "vue";
import Chatbox from "../components/Chatbox.vue";
import Settings from "../components/Settings.vue";
import Sidebar from "../components/Sidebar.vue";
import { getServerConfig } from "../apis";
import { useConfig, useSettings, useSystemState } from "../store";

const settings = useSettings();
const state = useSystemState();
const config = useConfig();

onMounted(async () => {
  state.testBackendConnection();

  // 1. If config store has a default model, apply it
  if (settings.model === null && config.data.defaultModel) {
    settings.setModel(config.data.defaultModel);
  }

  // 2. Fall back to server's env config for all settings
  try {
    const serverConfig = await getServerConfig();

    // Model
    if (settings.model === null && serverConfig.model !== null) {
      settings.setModel(serverConfig.model);
    }

    // VLM — only apply if no local VLM model configured
    if (config.data.vlm.model === null && serverConfig.vlm !== null) {
      config.setVlm({
        model: serverConfig.vlm.name,
        api_key: null,
        base_url: serverConfig.vlm.base_url,
        useMainModel: false,
      });
    }

    // Proxy
    if (config.data.proxy === null && serverConfig.proxy !== null) {
      config.setProxy(serverConfig.proxy);
    }

    // KB autosave
    if (!config.data.kbAutosave && serverConfig.kb_autosave) {
      config.setKbAutosave(serverConfig.kb_autosave);
    }
  } catch {
    // Server may not be reachable yet; warning will show and resolve on retry.
  }
});

const chatbox = ref<InstanceType<typeof Chatbox> | null>(null);

const settingsRef = ref<InstanceType<typeof Settings> | null>(null);

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

    <!-- Settings Modal -->
    <Dialog v-model:visible="state.showSettings" modal header="Settings" class="min-h-1/2 min-w-2xl">
      <Settings ref="settingsRef" @submit="state.showSettings = false" />
    </Dialog>

    <ConfirmPopup />
  </div>
</template>
