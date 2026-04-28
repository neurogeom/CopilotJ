<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { onMounted, ref } from "vue";
import { useSettings, useSystemState } from "../store";

type ChatboxExpose = {
  reset(): void;
  scrollToPost(postId: string): void;
};

const settings = useSettings();
const state = useSystemState();

const chatbox = ref<ChatboxExpose | null>(null);

onMounted(() => {
  void settings.loadServerModel();
});

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
    <Dialog :visible="state.showSettings" @update:visible="(value) => (state.showSettings = value)" modal header="Settings" class="min-h-1/2">
      <Settings @submit="state.showSettings = false" />
    </Dialog>

    <ConfirmPopup />
  </div>
</template>
