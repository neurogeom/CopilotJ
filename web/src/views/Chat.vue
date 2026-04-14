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
import { useSettings, useSystemState } from "../store";

const settings = useSettings();
const state = useSystemState();

onMounted(async () => {
  // Pre-populate the model from the server's env config so the "no model
  // configured" warning is suppressed when a model is already set server-side.
  if (settings.model === null) {
    try {
      const serverConfig = await getServerConfig();
      if (serverConfig.model !== null) {
        settings.setModel(serverConfig.model);
      }
    } catch {
      // Server may not be reachable yet; warning will show and resolve on retry.
    }
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
    <Dialog v-model:visible="state.showSettings" modal header="Settings" class="min-h-1/2">
      <Settings ref="settingsRef" @submit="state.showSettings = false" />
    </Dialog>

    <ConfirmPopup />
  </div>
</template>
