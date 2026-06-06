<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed } from "vue";
import Logo from "./Logo.vue";
import { isApiBaseConfigurable, isApiBaseConfigured } from "../apis/base";
import { useSettings, useSystemState } from "../store";

const settings = useSettings();
const state = useSystemState();

const showApiBaseWarning = computed(() => isApiBaseConfigurable && !isApiBaseConfigured());

const showConnectionWarning = computed(
  () => isApiBaseConfigurable && state.backendReachable === false && !state.connectionWarningDismissed,
);

const suggestions = [
  "Find best segmentation method for this image",
  "Segment this low signal-to-noise cell image",
  "Show my workflows",
  "Compare these two analysis methods",
];

const emit = defineEmits<{
  (event: "usePromptSuggestion", suggestion: string): void;
}>();

function usePromptSuggestion(suggestion: string) {
  emit("usePromptSuggestion", suggestion);
}
</script>

<template>
  <div class="my-20 flex flex-col items-center justify-center h-full text-center">
    <Logo class="w-24 h-24" />

    <h2 class="my-4 text-2xl font-semibold">CopilotJ</h2>

    <p class="text-sm max-w-md">Ask questions or give tasks for the Leader Agent to process.</p>

    <div
      v-if="settings.model === null"
      class="mt-6 px-4 py-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 text-sm text-amber-800 dark:text-amber-300 max-w-md"
    >
      No model configured. Click
      <button
        class="underline font-medium hover:text-amber-600 dark:hover:text-amber-200"
        @click="state.showSettings = true"
      >
        Settings
      </button>
      to set up a model and API key before submitting.
    </div>

    <div
      v-if="showApiBaseWarning"
      class="mt-2 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-sm text-red-800 dark:text-red-300 max-w-md"
    >
      API server URL is not configured. Click
      <button
        class="underline font-medium hover:text-red-600 dark:hover:text-red-200"
        @click="state.showSettings = true"
      >
        Settings
      </button>
      to configure a server URL.
    </div>

    <div
      v-if="showConnectionWarning"
      class="mt-2 px-4 py-3 rounded-lg bg-orange-50 dark:bg-orange-950/30 border border-orange-200 dark:border-orange-800 text-sm text-orange-800 dark:text-orange-300 max-w-md flex items-start justify-between gap-2"
    >
      <span
        >Cannot connect to the backend server. Please check if the API Server URL is correct in
        <button
          class="underline font-medium hover:text-orange-600 dark:hover:text-orange-200"
          @click="state.showSettings = true"
        >
          Settings
        </button>
        .</span
      >
      <button
        class="shrink-0 font-bold hover:text-orange-600 dark:hover:text-orange-200"
        @click="state.connectionWarningDismissed = true"
      >
        &times;
      </button>
    </div>

    <div class="w-full mt-8 grid md:grid-cols-2 grid-cols-1 gap-4">
      <div
        v-for="suggestion in suggestions"
        :key="suggestion"
        class="rounded-lg p-4 text-left bg-white dark:bg-gray-900 transition shadow-sm hover:shadow-md cursor-pointer"
        @click="usePromptSuggestion(suggestion)"
      >
        {{ suggestion }}
      </div>
    </div>
  </div>
</template>
