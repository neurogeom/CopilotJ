<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref, watch } from "vue";
import { setApiBaseUrl, testApiConnection } from "../apis/base";
import { getServerConfig } from "../apis";
import type { ServerConfig } from "../apis";

const props = defineProps<{
  apiBaseUrl: string;
  connectionStatus: "idle" | "testing" | "ok" | "fail";
}>();

const emit = defineEmits<{
  (e: "update:apiBaseUrl", value: string): void;
  (e: "update:connectionStatus", value: "idle" | "testing" | "ok" | "fail"): void;
  (e: "update:serverConfig", value: ServerConfig | null): void;
}>();

const apiBaseUrl = ref(props.apiBaseUrl);

// Sync from prop (fixes pre-fill when parent sets value async)
watch(
  () => props.apiBaseUrl,
  (val) => {
    apiBaseUrl.value = val;
  },
);

async function testConnection() {
  emit("update:connectionStatus", "testing");
  const ok = await testApiConnection(apiBaseUrl.value);
  emit("update:connectionStatus", ok ? "ok" : "fail");
  if (ok) {
    emit("update:apiBaseUrl", apiBaseUrl.value);
    try {
      setApiBaseUrl(apiBaseUrl.value);
      const config = await getServerConfig();
      emit("update:serverConfig", config);
    } catch {
      // Server may not return config
    }
  }
}
</script>

<template>
  <div class="flex flex-col gap-6">
    <p class="text-sm text-slate-500 dark:text-slate-400">Enter the URL of your CopilotJ server to get started.</p>

    <FormItem for="apiBaseUrl" label="Server URL" required>
      <div class="flex items-center gap-2">
        <InputText
          type="text"
          v-model="apiBaseUrl"
          inputId="apiBaseUrl"
          placeholder="http://localhost:8786"
          class="w-full"
          @keyup.enter="testConnection"
        />
        <Button label="Connect" :loading="connectionStatus === 'testing'" @click="testConnection" />
      </div>
      <p v-if="connectionStatus === 'ok'" class="text-sm text-green-600 dark:text-green-400 mt-1">
        Connected successfully
      </p>
      <p v-else-if="connectionStatus === 'fail'" class="text-sm text-red-600 dark:text-red-400 mt-1">
        Could not reach the server. Please check the URL and try again.
      </p>
    </FormItem>
  </div>
</template>
