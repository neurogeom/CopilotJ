<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { reactive, ref } from "vue";
import type { ServerConfig, ThreadConfigModel } from "../apis";
import { setApiBaseUrl } from "../apis/base";
import { useConfig, useSettings } from "../store";
import SettingsConnection from "../components/SettingsConnection.vue";
import SettingsLLM from "../components/SettingsLLM.vue";
import SettingsVLM from "../components/SettingsVLM.vue";
import SettingsAdvanced from "../components/SettingsAdvanced.vue";
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
  model: null as ThreadConfigModel | null,
  vlm: {
    useMainModel: true,
    model: null as string | null,
    apiKey: null as string | null,
    baseUrl: null as string | null,
  },
  proxy: null as string | null,
  tavilyApiKey: null as string | null,
  kbAutosave: false,
  autoScroll: true,
});

// Refs to child components for reading values
const connectionRef = ref<InstanceType<typeof SettingsConnection> | null>(null);
const llmRef = ref<InstanceType<typeof SettingsLLM> | null>(null);
const vlmRef = ref<InstanceType<typeof SettingsVLM> | null>(null);
const advancedRef = ref<InstanceType<typeof SettingsAdvanced> | null>(null);

function completeSetup() {
  setApiBaseUrl(wizard.apiBaseUrl);
  config.setDefaultModel(wizard.model);
  config.setVlm({
    model: wizard.vlm.model,
    api_key: wizard.vlm.apiKey,
    base_url: wizard.vlm.baseUrl,
    useMainModel: wizard.vlm.useMainModel,
  });
  config.setProxy(wizard.proxy);
  config.setTavilyApiKey(wizard.tavilyApiKey);
  config.setKbAutosave(wizard.kbAutosave);
  settings.setModel(wizard.model);
  settings.toggleAutoScroll(wizard.autoScroll);
  emit("complete");
}
</script>

<template>
  <Stepper value="1" linear class="setup-stepper">
    <StepList>
      <Step value="1">Connection</Step>
      <Step value="2">Model</Step>
      <Step value="3">Vision</Step>
      <Step value="4">Advanced</Step>
      <Step value="5">Finish</Step>
    </StepList>

    <StepPanels>
      <!-- Step 1: Connection -->
      <StepPanel v-slot="{ activateCallback }" value="1">
        <SettingsConnection
          ref="connectionRef"
          :api-base-url="wizard.apiBaseUrl"
          :connection-status="wizard.connectionStatus"
          @update:api-base-url="wizard.apiBaseUrl = $event"
          @update:connection-status="wizard.connectionStatus = $event"
          @update:server-config="wizard.serverConfig = $event"
        />
        <div class="flex pt-4 justify-end mt-auto">
          <Button label="Next" :disabled="wizard.connectionStatus !== 'ok'" @click="activateCallback('2')" />
        </div>
      </StepPanel>

      <!-- Step 2: Model -->
      <StepPanel v-slot="{ activateCallback }" value="2">
        <SettingsLLM
          ref="llmRef"
          :model="wizard.model"
          :server-config="wizard.serverConfig"
          @update:model="wizard.model = $event"
        />
        <div class="flex pt-4 justify-between mt-auto">
          <Button label="Back" severity="secondary" @click="activateCallback('1')" />
          <Button
            label="Next"
            :disabled="!llmRef?.isValid"
            @click="
              wizard.model = llmRef?.getModelValue() ?? null;
              activateCallback('3');
            "
          />
        </div>
      </StepPanel>

      <!-- Step 3: Vision -->
      <StepPanel v-slot="{ activateCallback }" value="3">
        <SettingsVLM
          ref="vlmRef"
          :use-main-model="wizard.vlm.useMainModel"
          :model="wizard.vlm.model"
          :api-key="wizard.vlm.apiKey"
          :base-url="wizard.vlm.baseUrl"
          @update="wizard.vlm = { ...wizard.vlm, ...$event }"
        />
        <div class="flex pt-4 justify-between mt-auto">
          <Button label="Back" severity="secondary" @click="activateCallback('2')" />
          <Button
            label="Next"
            :disabled="!vlmRef?.isValid"
            @click="
              wizard.vlm = { ...wizard.vlm, ...vlmRef!.getVlmValue() };
              activateCallback('4');
            "
          />
        </div>
      </StepPanel>

      <!-- Step 4: Advanced -->
      <StepPanel v-slot="{ activateCallback }" value="4">
        <SettingsAdvanced
          ref="advancedRef"
          :proxy="wizard.proxy"
          :tavily-api-key="wizard.tavilyApiKey"
          :kb-autosave="wizard.kbAutosave"
          :auto-scroll="wizard.autoScroll"
          @update="
            wizard.proxy = $event.proxy;
            wizard.tavilyApiKey = $event.tavilyApiKey;
            wizard.kbAutosave = $event.kbAutosave;
            wizard.autoScroll = $event.autoScroll;
          "
        />
        <div class="flex pt-4 justify-between mt-auto">
          <Button label="Back" severity="secondary" @click="activateCallback('3')" />
          <div class="flex gap-2">
            <Button label="Skip" severity="secondary" outlined @click="activateCallback('5')" />
            <Button label="Next" @click="activateCallback('5')" />
          </div>
        </div>
      </StepPanel>

      <!-- Step 5: Summary -->
      <StepPanel v-slot="{ activateCallback }" value="5">
        <SettingsSummary :wizard-data="wizard" @complete="completeSetup" @back="activateCallback('4')" />
      </StepPanel>
    </StepPanels>
  </Stepper>
</template>

<style scoped>
.setup-stepper {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.setup-stepper :deep(.p-stepper-panels) {
  flex: 1;
  display: flex;
}

.setup-stepper :deep(.p-stepper-panel) {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.setup-stepper :deep(.p-stepper-panel > div) {
  flex: 1;
  display: flex;
  flex-direction: column;
}
</style>
