<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { useDebounceFn } from "@vueuse/core";
import { isExplicit } from "../apis";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import { setApiBaseUrl } from "../apis/base";
import {
  checkVisionResolvable,
  toVlmConfig,
  validateAgreement,
  validateBase,
  validateModel,
  validateVlm,
  type ValidationInput,
} from "../lib";
import { useConfig, useSettings } from "../store";
import SettingsConnection from "../components/SettingsConnection.vue";
import UsageNotice from "../components/UsageNotice.vue";
import SettingsModel from "../components/SettingsModel.vue";
import SettingsVLM from "../components/SettingsVLM.vue";
import SettingsPreference from "../components/SettingsPreference.vue";
import SettingsSummary from "../components/SettingsSummary.vue";

const emit = defineEmits<{
  (e: "complete"): void;
}>();

const config = useConfig();
const settings = useSettings();

const wizard = reactive({
  apiBaseUrl: "",
  connectionStatus: "idle" as "idle" | "testing" | "ok" | "fail",
  serverConfig: null as ServerConfig | null,
  model: { use_server: true } as ThreadConfigModel,
  vlm: {
    useMainModel: true,
    model: null as string | null,
    apiKey: null as string | null,
    baseUrl: null as string | null,
    provider: null as string | null,
    visionEnabled: false,
  },
  // Notice step state (committed in completeSetup); seeded from any existing config.
  userAgreement: config.data.userAgreement,
  visionEnabled: config.data.visionEnabled ?? config.serverVisionEnabled ?? false,
  proxy: null as string | null,
  tavilyApiKey: null as string | null,
  kbAutosave: false,
  autoScroll: true,
});

// Effective main model name for the Vision capability check: the explicit
// choice, or the server's model name when "use server" is selected.
const mainModelName = computed(() => {
  const m = wizard.model;
  return m && isExplicit(m) ? m.name : (wizard.serverConfig?.model?.name ?? null);
});

// Shared validation (../lib) — the same rules the Settings dialog uses. Each
// step gates on its own concern; values flow live via v-model, so no child refs
// or getValue() reads are needed.
function wizardInput(): ValidationInput {
  return {
    userAgreement: wizard.userAgreement,
    visionEnabled: wizard.visionEnabled,
    model: wizard.model,
    vlm: toVlmConfig(wizard.vlm),
    connectionStatus: wizard.connectionStatus,
    apiBaseUrl: wizard.apiBaseUrl,
  };
}
const agreementOk = computed(() => validateAgreement(wizardInput()).ok);
const baseOk = computed(() => validateBase(wizardInput()).ok);
const modelOk = computed(() => validateModel(wizardInput()).ok);
const vlmOk = computed(() => validateVlm(wizardInput()).ok);

// Vision resolvability (async) — a safety net on top of vlmOk. It covers the
// window before SettingsVLM's internal capability check settles, where "use main
// model" could still be on for a main model that lacks vision. Only a confirmed
// "no vision" result blocks; unknown models and lookup failures are allowed.
const visionResolvable = ref(true);
async function refreshVisionResolvable() {
  if (!wizard.visionEnabled) {
    visionResolvable.value = true;
    return;
  }
  const result = await checkVisionResolvable({
    visionEnabled: true,
    mainModelName: mainModelName.value,
    vlm: toVlmConfig(wizard.vlm),
  });
  visionResolvable.value = result.ok;
}
const debouncedRefreshVision = useDebounceFn(refreshVisionResolvable, 400);
watch([mainModelName, () => wizard.vlm, () => wizard.visionEnabled], () => debouncedRefreshVision(), {
  deep: true,
  immediate: true,
});

// v-model proxies: the components use camelCase composite values; map them onto
// the wizard's reactive fields so edits are live-synced (committed on Finish).
const wizardVlm = computed<{
  useMainModel: boolean;
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  provider: string | null;
}>({
  get: () => ({
    useMainModel: wizard.vlm.useMainModel,
    model: wizard.vlm.model,
    apiKey: wizard.vlm.apiKey,
    baseUrl: wizard.vlm.baseUrl,
    provider: wizard.vlm.provider,
  }),
  set: (v) => {
    wizard.vlm.useMainModel = v.useMainModel;
    wizard.vlm.model = v.model;
    wizard.vlm.apiKey = v.apiKey;
    wizard.vlm.baseUrl = v.baseUrl;
    wizard.vlm.provider = v.provider;
  },
});
const wizardPref = computed<{
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  autoScroll: boolean;
}>({
  get: () => ({
    proxy: wizard.proxy,
    tavilyApiKey: wizard.tavilyApiKey,
    kbAutosave: wizard.kbAutosave,
    autoScroll: wizard.autoScroll,
  }),
  set: (v) => {
    wizard.proxy = v.proxy;
    wizard.tavilyApiKey = v.tavilyApiKey;
    wizard.kbAutosave = v.kbAutosave;
    wizard.autoScroll = v.autoScroll;
  },
});

function completeSetup() {
  setApiBaseUrl(wizard.apiBaseUrl);
  config.setDefaultModel(wizard.model);
  // Populate the runtime server model (Chat.vue skips its onMounted fetch during the wizard).
  config.setServerModel(wizard.serverConfig?.model ?? null);
  config.setVlm({
    model: wizard.vlm.model,
    api_key: wizard.vlm.apiKey,
    base_url: wizard.vlm.baseUrl,
    provider: wizard.vlm.provider,
    useMainModel: wizard.vlm.useMainModel,
  });
  // Notice step: commit the agreement + Vision opt-in (edit-then-save, like the other steps).
  config.setUserAgreement(wizard.userAgreement);
  config.setVisionEnabled(wizard.visionEnabled);
  config.setProxy(wizard.proxy);
  config.setTavilyApiKey(wizard.tavilyApiKey);
  config.setKbAutosave(wizard.kbAutosave);
  // Active model: the user's choice ({use_server:true} or an explicit model).
  settings.setModel(wizard.model);
  settings.toggleAutoScroll(wizard.autoScroll);
  emit("complete");
}
</script>

<template>
  <Stepper value="1" linear class="setup-stepper">
    <StepList>
      <Step value="1">Notice</Step>
      <Step value="2">Connection</Step>
      <Step value="3">Model</Step>
      <!-- Always listed in the progress bar; auto-skipped when Vision is disabled. -->
      <Step value="4">Vision</Step>
      <Step value="5">Preferences</Step>
      <Step value="6">Finish</Step>
    </StepList>

    <StepPanels>
      <!-- Step 1: Notice (User Agreement + Vision opt-in) -->
      <StepPanel v-slot="{ activateCallback }" value="1">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="flex flex-col gap-6 min-h-0 flex-1 overflow-y-auto">
            <UsageNotice v-model:userAgreement="wizard.userAgreement" v-model:visionEnabled="wizard.visionEnabled" />
          </div>
          <div class="flex pt-4 justify-end">
            <Button label="Next" :disabled="!agreementOk" @click="activateCallback('2')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 2: Connection -->
      <StepPanel v-slot="{ activateCallback }" value="2">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsConnection
              v-model:api-base-url="wizard.apiBaseUrl"
              v-model:connection-status="wizard.connectionStatus"
              @update:server-config="wizard.serverConfig = $event"
            />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('1')" />
            <Button label="Next" :disabled="!baseOk" @click="activateCallback('3')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 3: Model -->
      <StepPanel v-slot="{ activateCallback }" value="3">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsModel v-model="wizard.model" :server-config="wizard.serverConfig" />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('2')" />
            <Button label="Next" :disabled="!modelOk" @click="activateCallback(wizard.visionEnabled ? '4' : '5')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 4: Vision — shown in the bar always; panel only when Vision is enabled. -->
      <StepPanel v-if="wizard.visionEnabled" v-slot="{ activateCallback }" value="4">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsVLM v-model="wizardVlm" :main-model-name="mainModelName" />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('3')" />
            <Button label="Next" :disabled="!vlmOk || !visionResolvable" @click="activateCallback('5')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 5: Preferences -->
      <StepPanel v-slot="{ activateCallback }" value="5">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsPreference v-model="wizardPref" />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback(wizard.visionEnabled ? '4' : '3')" />
            <div class="flex gap-2">
              <Button label="Skip" severity="secondary" outlined @click="activateCallback('6')" />
              <Button label="Next" @click="activateCallback('6')" />
            </div>
          </div>
        </div>
      </StepPanel>

      <!-- Step 6: Summary -->
      <StepPanel v-slot="{ activateCallback }" value="6">
        <SettingsSummary :wizard-data="wizard" @complete="completeSetup" @back="activateCallback('5')" />
      </StepPanel>
    </StepPanels>
  </Stepper>
</template>

<style scoped>
.setup-stepper {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  width: 100%;
}

.setup-stepper :deep(.p-steppanels) {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.setup-stepper :deep(.p-steppanel) {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.setup-stepper :deep(.p-steppanel > div) {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
</style>
