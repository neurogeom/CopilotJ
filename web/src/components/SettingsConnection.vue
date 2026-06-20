<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref, watch } from "vue";
import { setApiBaseUrl, testApiConnection } from "../apis/base";
import { getServerConfig } from "../apis";
import type { ServerConfig } from "../apis";

const props = withDefaults(
  defineProps<{
    apiBaseUrl: string;
    /** Initial status; the component owns status afterwards and emits updates. */
    connectionStatus?: "idle" | "testing" | "ok" | "fail";
    /** Settings: persist + reload the page on a successful connect. Wizard: fetch config instead. */
    reloadOnConnect?: boolean;
    /** Show the inline Connect/Save button. Hidden in Settings (the pinned footer triggers connect); shown in the wizard. */
    showConnectButton?: boolean;
  }>(),
  { connectionStatus: "idle", reloadOnConnect: false, showConnectButton: true },
);

const emit = defineEmits<{
  (e: "update:apiBaseUrl", value: string): void;
  (e: "update:connectionStatus", value: "idle" | "testing" | "ok" | "fail"): void;
  (e: "update:serverConfig", value: ServerConfig | null): void;
}>();

const apiBaseUrl = ref(props.apiBaseUrl);
const status = ref<"idle" | "testing" | "ok" | "fail">(props.connectionStatus);

// Sync from prop (fixes pre-fill when parent sets value async)
watch(
  () => props.apiBaseUrl,
  (val) => {
    apiBaseUrl.value = val;
  },
);

function setStatus(value: "idle" | "testing" | "ok" | "fail") {
  status.value = value;
  emit("update:connectionStatus", value);
}

async function connect() {
  setStatus("testing");
  const ok = await testApiConnection(apiBaseUrl.value);
  setStatus(ok ? "ok" : "fail");
  if (!ok) return;

  emit("update:apiBaseUrl", apiBaseUrl.value);
  try {
    setApiBaseUrl(apiBaseUrl.value);
  } catch {
    // ignore persistence errors
  }

  // Settings changes the backend mid-session: reload to re-initialise.
  // Wizard keeps going: fetch the server config so later steps can pre-fill.
  if (props.reloadOnConnect) {
    window.location.reload();
    return;
  }
  try {
    const config = await getServerConfig();
    emit("update:serverConfig", config);
  } catch {
    // Server may not return config
  }
}

function getValue() {
  return apiBaseUrl.value;
}

defineExpose({ getValue, connect });
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
          @keyup.enter="connect"
        />
        <Button
          v-if="showConnectButton"
          :label="reloadOnConnect ? 'Save' : 'Connect'"
          :loading="status === 'testing'"
          @click="connect"
        />
      </div>
      <p v-if="status === 'ok'" class="text-sm text-green-600 dark:text-green-400 mt-1">Connected successfully</p>
      <p v-else-if="status === 'fail'" class="text-sm text-red-600 dark:text-red-400 mt-1">
        Could not reach the server. Please check the URL and try again.
      </p>
    </FormItem>
  </div>
</template>
