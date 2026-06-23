<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref, watch } from "vue";
import { setApiBaseUrl, testApiConnection } from "../apis/base";
import { getServerConfig } from "../apis";
import type { ServerConfig } from "../apis";

type ConnectionStatus = "idle" | "testing" | "ok" | "fail";

const props = withDefaults(
  defineProps<{
    /** Settings: persist + reload the page on a successful connect. Wizard: fetch config instead. */
    reloadOnConnect?: boolean;
    /** Show the inline Connect/Save button. Hidden in Settings (the pinned footer triggers connect); shown in the wizard. */
    showConnectButton?: boolean;
  }>(),
  { reloadOnConnect: false, showConnectButton: true },
);

const emit = defineEmits<{
  (e: "update:serverConfig", value: ServerConfig | null): void;
}>();

// Two-way models: the parent (Settings draft / Wizard reactive) owns these and
// tracks live edits. apiBaseUrl emits on every keystroke; connectionStatus on
// every status change. (defineModel registers the update: events itself.)
const apiBaseUrl = defineModel<string>("apiBaseUrl", { required: true });
const connectionStatus = defineModel<ConnectionStatus>("connectionStatus", { default: "idle" });

// The URL at the last "ok". Lets us demote a stale "Connected successfully" the
// moment the user edits the URL — so the status never lies about an untested
// URL. Kept in sync whether "ok" arrives from a successful connect here or from
// the parent seeding it (Settings reopens already-connected to the running
// server, so the unchanged URL stays valid without a re-test).
const lastOkUrl = ref(apiBaseUrl.value);

function setStatus(value: ConnectionStatus) {
  connectionStatus.value = value;
}

// Remember the URL whenever the connection is marked healthy.
watch(connectionStatus, (s) => {
  if (s === "ok") lastOkUrl.value = apiBaseUrl.value;
});

// Editing the URL after a successful connect invalidates that connection.
watch(apiBaseUrl, (v) => {
  if (connectionStatus.value === "ok" && v !== lastOkUrl.value) setStatus("idle");
});

async function connect() {
  setStatus("testing");
  const ok = await testApiConnection(apiBaseUrl.value);
  setStatus(ok ? "ok" : "fail");
  if (!ok) return;

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
          :loading="connectionStatus === 'testing'"
          @click="connect"
        />
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
