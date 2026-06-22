<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { isExplicit } from "../apis";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import { setApiBaseUrl } from "../apis/base";
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
  visionEnabled: config.data.visionEnabled,
  proxy: null as string | null,
  tavilyApiKey: null as string | null,
  kbAutosave: false,
  autoScroll: true,
});

// Refs to child components for reading values
const connectionRef = ref<InstanceType<typeof SettingsConnection> | null>(null);
const llmRef = ref<InstanceType<typeof SettingsModel> | null>(null);
const vlmRef = ref<InstanceType<typeof SettingsVLM> | null>(null);
const advancedRef = ref<InstanceType<typeof SettingsPreference> | null>(null);

// Effective main model name for the Vision capability check: the explicit
// choice, or the server's model name when "use server" is selected.
const mainModelName = computed(() => {
  const m = wizard.model;
  return m && isExplicit(m) ? m.name : (wizard.serverConfig?.model?.name ?? null);
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
            <Button label="Next" :disabled="!wizard.userAgreement" @click="activateCallback('2')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 2: Connection -->
      <StepPanel v-slot="{ activateCallback }" value="2">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsConnection
              ref="connectionRef"
              :api-base-url="wizard.apiBaseUrl"
              :connection-status="wizard.connectionStatus"
              @update:api-base-url="wizard.apiBaseUrl = $event"
              @update:connection-status="wizard.connectionStatus = $event"
              @update:server-config="wizard.serverConfig = $event"
            />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('1')" />
            <Button label="Next" :disabled="wizard.connectionStatus !== 'ok'" @click="activateCallback('3')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 3: Model -->
      <StepPanel v-slot="{ activateCallback }" value="3">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsModel
              ref="llmRef"
              :model="wizard.model"
              :server-config="wizard.serverConfig"
              @update:model="wizard.model = $event"
            />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('2')" />
            <Button
              label="Next"
              :disabled="!llmRef?.isValid"
              @click="
                wizard.model = llmRef?.getModelValue() ?? { use_server: true };
                activateCallback(wizard.visionEnabled ? '4' : '5');
              "
            />
          </div>
        </div>
      </StepPanel>

      <!-- Step 4: Vision — shown in the bar always; panel only when Vision is enabled. -->
      <StepPanel v-if="wizard.visionEnabled" v-slot="{ activateCallback }" value="4">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsVLM
              ref="vlmRef"
              :use-main-model="wizard.vlm.useMainModel"
              :model="wizard.vlm.model"
              :api-key="wizard.vlm.apiKey"
              :base-url="wizard.vlm.baseUrl"
              :provider="wizard.vlm.provider"
              :main-model-name="mainModelName"
              @update="wizard.vlm = { ...wizard.vlm, ...$event }"
            />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback('3')" />
            <Button
              label="Next"
              :disabled="!vlmRef?.isValid"
              @click="
                wizard.vlm = { ...wizard.vlm, ...vlmRef!.getVlmValue() };
                activateCallback('5');
              "
            />
          </div>
        </div>
      </StepPanel>

      <!-- Step 5: Preferences -->
      <StepPanel v-slot="{ activateCallback }" value="5">
        <div class="flex min-h-0 flex-1 flex-col">
          <div class="min-h-0 flex-1 overflow-y-auto">
            <SettingsPreference
              ref="advancedRef"
              :proxy="wizard.proxy"
              :tavily-api-key="wizard.tavilyApiKey"
              :kb-autosave="wizard.kbAutosave"
              :auto-scroll="wizard.autoScroll"
            />
          </div>
          <div class="flex pt-4 justify-between">
            <Button label="Back" severity="secondary" @click="activateCallback(wizard.visionEnabled ? '4' : '3')" />
            <div class="flex gap-2">
              <Button label="Skip" severity="secondary" outlined @click="activateCallback('6')" />
              <Button
                label="Next"
                @click="
                  wizard.proxy = advancedRef?.getValue().proxy ?? null;
                  wizard.tavilyApiKey = advancedRef?.getValue().tavilyApiKey ?? null;
                  wizard.kbAutosave = advancedRef?.getValue().kbAutosave ?? false;
                  wizard.autoScroll = advancedRef?.getValue().autoScroll ?? true;
                  activateCallback('6');
                "
              />
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
