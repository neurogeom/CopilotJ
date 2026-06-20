<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref } from "vue";
import type { ThreadConfigModel } from "../apis";
import { getBaseUrl } from "../apis/base";
import { useConfig, useSettings } from "../store";
import SettingsConnection from "./SettingsConnection.vue";
import SettingsModel from "./SettingsModel.vue";
import SettingsVLM from "./SettingsVLM.vue";
import SettingsPreference from "./SettingsPreference.vue";

const settings = useSettings();
const config = useConfig();

// Base URL tab seeds from the current API base URL; the component persists + reloads on connect.
const apiBaseUrl = ref(getBaseUrl().replace(/\/api$/, ""));

const activeTab = ref("base");
const connectionRef = ref<InstanceType<typeof SettingsConnection> | null>(null);
const modelRef = ref<InstanceType<typeof SettingsModel> | null>(null);
const vlmRef = ref<InstanceType<typeof SettingsVLM> | null>(null);
const prefRef = ref<InstanceType<typeof SettingsPreference> | null>(null);

function submitModel(model: ThreadConfigModel | null) {
  if (model === null) {
    // useDefaultModel = true: don't persist, use server model for this session
    config.setDefaultModel(null);
    settings.setModel(config.serverModel);
  } else {
    // User-configured model: persist to localStorage
    config.setDefaultModel(model);
    settings.setModel(model);
  }
}

// --- VLM config ---
function submitVlm(vlm: {
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  provider: string | null;
  useMainModel: boolean;
  visionEnabled: boolean;
}) {
  config.setVlm({
    model: vlm.model,
    api_key: vlm.apiKey,
    base_url: vlm.baseUrl,
    provider: vlm.provider,
    useMainModel: vlm.useMainModel,
  });
  config.setVisionEnabled(vlm.visionEnabled);
}

// --- Preferences (proxy, Tavily, KB autosave, auto-scroll) ---
function savePreference(value: {
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  autoScroll: boolean;
}) {
  config.setProxy(value.proxy);
  config.setTavilyApiKey(value.tavilyApiKey);
  config.setKbAutosave(value.kbAutosave);
  settings.toggleAutoScroll(value.autoScroll);
}

function onSubmit() {
  switch (activeTab.value) {
    case "base":
      connectionRef.value?.connect();
      break;
    case "model":
      if (modelRef.value) submitModel(modelRef.value.getModelValue());
      break;
    case "vision":
      if (vlmRef.value) submitVlm(vlmRef.value.getVlmValue());
      break;
    case "pref":
      if (prefRef.value) savePreference(prefRef.value.getValue());
      break;
  }
}
</script>

<template>
  <div class="flex min-h-0 w-full flex-1 flex-col">
    <Tabs v-model:value="activeTab" class="flex min-h-0 flex-1 flex-col">
      <TabList>
        <Tab value="base">Base</Tab>
        <Tab value="model">Model</Tab>
        <Tab value="vision">Vision</Tab>
        <Tab value="pref">Preferences</Tab>
      </TabList>

      <TabPanels class="flex-1 overflow-y-auto">
        <!-- Base Tab -->
        <TabPanel value="base">
          <SettingsConnection
            ref="connectionRef"
            :api-base-url="apiBaseUrl"
            :reload-on-connect="true"
            :show-connect-button="false"
          />
        </TabPanel>

        <!-- Model Tab -->
        <TabPanel value="model">
          <SettingsModel
            ref="modelRef"
            :model="settings.model"
            :server-model-name="config.serverModel?.name ?? null"
            :show-submit-button="false"
          />
        </TabPanel>

        <!-- Vision Tab -->
        <TabPanel value="vision">
          <SettingsVLM
            ref="vlmRef"
            :model="config.data.vlm.model"
            :api-key="config.data.vlm.api_key"
            :base-url="config.data.vlm.base_url"
            :provider="config.data.vlm.provider"
            :use-main-model="config.data.vlm.useMainModel"
            :main-model-name="settings.model?.name ?? null"
            :vision-enabled="config.data.visionEnabled"
            :show-submit-button="false"
          />
        </TabPanel>

        <!-- Preferences Tab -->
        <TabPanel value="pref">
          <SettingsPreference
            ref="prefRef"
            :proxy="config.data.proxy"
            :tavily-api-key="config.data.tavilyApiKey"
            :kb-autosave="config.data.kbAutosave"
            :auto-scroll="settings.autoScroll"
            :show-submit-button="false"
          />
        </TabPanel>
      </TabPanels>
    </Tabs>

    <!-- Pinned submit (per active tab) -->
    <div class="flex justify-end pt-4">
      <Button label="Save" @click="onSubmit" />
    </div>
  </div>
</template>
