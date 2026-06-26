<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { IconAlertTriangle } from "@tabler/icons-vue";
import type { ThreadConfigModel } from "../apis";
import { getBaseUrl } from "../apis/base";
import {
  resolveMainModelName,
  toVlmConfig,
  validateBase,
  validateModel,
  validateVlm,
  validateVision,
  type ValidationInput,
  type ValidationKey,
} from "../lib";
import { useActiveThread, useConfig, useSettings, useSystemState } from "../store";
import type { SettingsTab } from "../store";
import SettingsConnection from "./SettingsConnection.vue";
import UsageNotice from "./UsageNotice.vue";
import SettingsModel from "./SettingsModel.vue";
import SettingsVLM from "./SettingsVision.vue";
import SettingsPreference from "./SettingsPreference.vue";

const settings = useSettings();
const config = useConfig();
const state = useSystemState();
const activeThread = useActiveThread();

// The dialog's working copy — edited live via v-model, committed to the store
// only on Save. Re-seeded from the committed config each time the dialog opens,
// so closing via the header × discards pending edits.
const draft = reactive<{
  userAgreement: boolean;
  visionEnabled: boolean;
  model: ThreadConfigModel | null;
  vlm: {
    useServerVlm: boolean;
    useMainModel: boolean;
    model: string | null;
    apiKey: string | null;
    baseUrl: string | null;
    provider: string | null;
  };
  pref: { proxy: string | null; tavilyApiKey: string | null; kbAutosave: boolean; autoScroll: boolean };
  apiBaseUrl: string;
  // Owned with SettingsConnection; "ok" only while the URL is connected & unchanged.
  connectionStatus: "idle" | "testing" | "ok" | "fail";
}>({
  userAgreement: false,
  visionEnabled: false,
  model: null,
  vlm: { useServerVlm: false, useMainModel: true, model: null, apiKey: null, baseUrl: null, provider: null },
  pref: { proxy: null, tavilyApiKey: null, kbAutosave: false, autoScroll: true },
  apiBaseUrl: getBaseUrl().replace(/\/api$/, ""),
  // The running server is already connected; an unchanged URL stays valid without
  // a re-test. Demoted to "idle" by SettingsConnection once the URL is edited.
  connectionStatus: "ok",
});

// Submit-time validation error (required fields / vision gate). Declared before
// seedDraft() because the immediate showSettings watch calls it during setup.
const formError = ref<string | null>(null);

function seedDraft() {
  draft.userAgreement = config.data.userAgreement;
  // visionEnabled is nullable on main (null = defer to server); coerce for the draft.
  draft.visionEnabled = config.data.visionEnabled ?? config.serverVisionEnabled ?? false;
  draft.model = config.data.defaultModel;
  draft.vlm = {
    useServerVlm: config.data.vlm.useServerVlm,
    useMainModel: config.data.vlm.useMainModel,
    model: config.data.vlm.model,
    apiKey: config.data.vlm.api_key,
    baseUrl: config.data.vlm.base_url,
    provider: config.data.vlm.provider,
  };
  draft.pref = {
    proxy: config.data.proxy,
    tavilyApiKey: config.data.tavilyApiKey,
    kbAutosave: config.data.kbAutosave,
    autoScroll: settings.autoScroll,
  };
  draft.apiBaseUrl = getBaseUrl().replace(/\/api$/, "");
  // Re-assume the running server is connected each time the dialog opens.
  draft.connectionStatus = "ok";
  formError.value = null;
}

// Seed whenever the dialog opens.
watch(
  () => state.showSettings,
  (open) => {
    if (open) seedDraft();
  },
  { immediate: true },
);

// The active tab lives in the store so cross-tab jumps and chat-error
// navigation land on the right tab.
const activeTab = computed<SettingsTab>({
  get: () => state.settingsTab,
  set: (v) => {
    state.settingsTab = v;
  },
});

const connectionRef = ref<InstanceType<typeof SettingsConnection> | null>(null);

// A thread (or an in-flight send/optimize) ties the app to the current backend,
// so the Base URL is read-only until the user starts a new thread. Covers the
// pre-start window too: status is still "init" while loading/optimizing.
const threadActive = computed(() => {
  const t = activeThread.thread;
  return t.status !== "init" || t.loading || t.optimizing;
});

// Effective main model name (used by the submit-time vision check below and by
// SettingsVLM's native capability check on the Vision tab).
const draftMainModelName = computed(() => resolveMainModelName(draft.model, config.serverModel));

// Build the normalized validation input from the draft. All rules live in the
// shared lib (../lib) so Settings and Wizard can never drift apart.
function draftInput(): ValidationInput {
  return {
    userAgreement: draft.userAgreement,
    visionEnabled: draft.visionEnabled,
    model: draft.model,
    vlm: toVlmConfig(draft.vlm),
    connectionStatus: draft.connectionStatus,
    apiBaseUrl: draft.apiBaseUrl,
    serverModelAvailable: !!config.serverModel,
    serverVlmAvailable: !!config.serverVlm,
  };
}

// Map a failing concern to the tab the user should fix it on.
function tabForKey(key: ValidationKey): SettingsTab {
  switch (key) {
    case "agreement":
      return "notice";
    case "base":
      return "base";
    case "model":
      return "model";
    case "vlm":
    case "vision":
      return "vision";
  }
}

// Clear any stale submit error as soon as the user edits the draft.
watch(draft, () => {
  formError.value = null;
});

// --- Commit the draft to the store on Save. ---
function commitDraft() {
  config.setUserAgreement(draft.userAgreement);
  config.setVisionEnabled(draft.visionEnabled);
  config.setDefaultModel(draft.model);
  settings.setModel(draft.model);
  config.setVlm({
    model: draft.vlm.model,
    api_key: draft.vlm.apiKey,
    base_url: draft.vlm.baseUrl,
    provider: draft.vlm.provider,
    useMainModel: draft.vlm.useMainModel,
    useServerVlm: draft.vlm.useServerVlm,
  });
  config.setProxy(draft.pref.proxy);
  config.setTavilyApiKey(draft.pref.tavilyApiKey);
  config.setKbAutosave(draft.pref.kbAutosave);
  settings.toggleAutoScroll(draft.pref.autoScroll);
  formError.value = null;
}

// Save is always clickable. Base is a pure connection action (test URL + reload
// on success) that does not commit the draft. Every other tab validates the
// WHOLE draft on click via the shared rules: synchronous concerns first
// (base → model → vlm), then the async vision gate. On any failure we show a
// message and jump to the offending tab instead of committing. Agreement is
// intentionally not checked here — it is a first-run concern (Wizard only).
async function onSubmit() {
  if (activeTab.value === "base") {
    if (threadActive.value) {
      formError.value =
        "The server URL can't be changed while a conversation is in progress. Start a new thread or reload the page to switch servers.";
      return;
    }
    connectionRef.value?.connect();
    return;
  }
  const input = draftInput();
  const failed = [validateBase(input), validateModel(input), validateVlm(input)].find((c) => !c.ok);
  if (failed) {
    formError.value = failed.message;
    activeTab.value = tabForKey(failed.key);
    return;
  }
  // Async: only a confirmed "no vision" result blocks; lookup failures skip (allow).
  const vision = await validateVision(input, draftMainModelName.value);
  if (!vision.ok) {
    formError.value = vision.message;
    activeTab.value = "vision";
    return;
  }
  commitDraft();
  state.showSettings = false;
}

// Soft reconnect (Settings Base tab): the URL was just persisted by
// SettingsConnection.connect(); re-sync server config from the new backend and
// refresh the reachability badge, without reloading the page. If /api/config
// fails, keep the dialog open with a warning instead of hiding the stale state.
async function onReconnect() {
  const ok = await config.applyServerConfig();
  state.testBackendConnection();
  if (ok) {
    state.showSettings = false;
  } else {
    formError.value = "Connected, but the server didn't return its config. Model availability may be inaccurate.";
  }
}
</script>

<template>
  <div class="flex min-h-0 w-full flex-1 flex-col">
    <!-- Submit-time validation error (required fields / vision gate). -->
    <div
      v-if="formError"
      class="mb-4 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-700 dark:bg-red-950/40 dark:text-red-300"
    >
      <IconAlertTriangle size="16" class="mt-0.5 shrink-0" />
      <span class="flex-1">{{ formError }}</span>
      <button
        type="button"
        class="shrink-0 font-bold hover:text-red-900 dark:hover:text-red-100"
        @click="formError = null"
      >
        &times;
      </button>
    </div>

    <Tabs v-model:value="activeTab" class="flex min-h-0 flex-1 flex-col">
      <TabList>
        <Tab value="notice">Notice</Tab>
        <Tab value="base">Base</Tab>
        <Tab value="model">Model</Tab>
        <Tab value="vision">Vision</Tab>
        <Tab value="pref">Preferences</Tab>
      </TabList>

      <TabPanels class="flex-1 overflow-y-auto">
        <!-- Notice Tab (User Agreement + Vision opt-in) — live-bound to the draft. -->
        <TabPanel value="notice">
          <UsageNotice v-model:userAgreement="draft.userAgreement" v-model:visionEnabled="draft.visionEnabled" />
        </TabPanel>

        <!-- Base Tab (connection action: test URL + reload on success) -->
        <TabPanel value="base">
          <SettingsConnection
            ref="connectionRef"
            v-model:api-base-url="draft.apiBaseUrl"
            v-model:connection-status="draft.connectionStatus"
            :reconnect-on-connect="true"
            :locked="threadActive"
            :show-connect-button="false"
            @reconnect="onReconnect"
          />
        </TabPanel>

        <!-- Model Tab -->
        <TabPanel value="model">
          <SettingsModel v-model="draft.model" :server-model-name="config.serverModel?.name ?? null" />
        </TabPanel>

        <!-- Vision Tab (configurable only when Vision is enabled on the Notice tab) -->
        <TabPanel value="vision">
          <SettingsVLM
            v-if="draft.visionEnabled"
            v-model="draft.vlm"
            :main-model-name="draftMainModelName"
            :server-vlm-name="config.serverVlm?.name ?? null"
          />
          <p v-else class="text-sm text-slate-500 dark:text-slate-400">
            Vision is currently disabled.
            <button
              type="button"
              class="cursor-pointer font-medium text-violet-600 underline hover:text-violet-500 dark:text-violet-400"
              @click="activeTab = 'notice'"
            >
              Enable it on the Notice tab
            </button>
            to configure a vision model.
          </p>
        </TabPanel>

        <!-- Preferences Tab -->
        <TabPanel value="pref">
          <SettingsPreference v-model="draft.pref" />
        </TabPanel>
      </TabPanels>
    </Tabs>

    <!-- Pinned submit: always clickable; validates on click. -->
    <div class="flex justify-end pt-4">
      <Button label="Save" @click="onSubmit" />
    </div>
  </div>
</template>
