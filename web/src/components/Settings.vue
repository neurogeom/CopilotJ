<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref } from "vue";
import type { ThreadConfigModel, ThreadConfigQuery } from "../apis";
import { getBaseUrl, isApiBaseConfigurable, setApiBaseUrl, testApiConnection } from "../apis/base";
import { useConfig, useSettings } from "../store";
import SettingLLM from "./SettingLLM.vue";
import SettingVLM from "./SettingVLM.vue";

const emit = defineEmits<{
  (e: "submit", model: ThreadConfigQuery | null): void;
}>();

const settings = useSettings();
const config = useConfig();

// --- API Base URL config ---
const apiBaseUrl = ref(getBaseUrl().replace(/\/api$/, ""));
const connectionStatus = ref<"idle" | "testing" | "ok" | "fail">("idle");

async function testConnection() {
  connectionStatus.value = "testing";
  const ok = await testApiConnection(apiBaseUrl.value);
  connectionStatus.value = ok ? "ok" : "fail";
}

function saveApiBaseUrl() {
  setApiBaseUrl(apiBaseUrl.value);
  window.location.reload();
}

function sumbmitModel(model: ThreadConfigModel | null) {
  settings.setModel(model);
  config.setDefaultModel(model);
  emit("submit", { model: settings.model });
}

// --- VLM config ---
function submitVlm(vlm: {
  model: string | null;
  apiKey: string | null;
  baseUrl: string | null;
  useMainModel: boolean;
}) {
  config.setVlm({
    model: vlm.model,
    api_key: vlm.apiKey,
    base_url: vlm.baseUrl,
    useMainModel: vlm.useMainModel,
  });
}

// --- Integrations ---
const proxy = ref(config.data.proxy || "");
const tavilyApiKey = ref(config.data.tavilyApiKey || "");

function saveIntegrations() {
  config.setProxy(proxy.value || null);
  config.setTavilyApiKey(tavilyApiKey.value || null);
}
</script>

<template>
  <Tabs value="model">
    <TabList>
      <Tab value="model">Model</Tab>
      <Tab value="vision">Vision</Tab>
      <Tab value="integrations">Integrations</Tab>
      <Tab value="pref">Preferences</Tab>
    </TabList>

    <TabPanels>
      <!-- Model Tab -->
      <TabPanel value="model">
        <SettingLLM :model="settings.model" @update:model="sumbmitModel" />
      </TabPanel>

      <!-- Vision Tab -->
      <TabPanel value="vision">
        <SettingVLM
          :model="config.data.vlm.model"
          :api-key="config.data.vlm.api_key"
          :base-url="config.data.vlm.base_url"
          :use-main-model="config.data.vlm.useMainModel"
          @update="submitVlm"
        />
      </TabPanel>

      <!-- Integrations Tab -->
      <TabPanel value="integrations">
        <div class="space-y-4">
          <FormItem for="proxy" label="HTTP Proxy">
            <InputText
              type="text"
              v-model="proxy"
              inputId="proxy"
              placeholder="http://127.0.0.1:8080 (optional)"
              class="w-full"
            />
          </FormItem>

          <FormItem for="tavilyApiKey" label="Tavily API Key">
            <InputText
              type="text"
              v-model="tavilyApiKey"
              inputId="tavilyApiKey"
              placeholder="tvly-xxxxxxxx (optional, for web search)"
              class="w-full"
            />
          </FormItem>

          <Button label="Save Integrations" @click="saveIntegrations" />
        </div>
      </TabPanel>

      <!-- Preferences Tab -->
      <TabPanel value="pref">
        <div class="space-y-4">
          <FormItem v-if="isApiBaseConfigurable" for="apiBaseUrl" label="API Server URL">
            <div class="flex items-center gap-2">
              <InputText
                type="text"
                v-model="apiBaseUrl"
                inputId="apiBaseUrl"
                placeholder="http://localhost:8786"
                class="w-full"
              />
              <Button class="w-24" label="Test" :loading="connectionStatus === 'testing'" @click="testConnection" />
              <Button class="w-24" label="Save" @click="saveApiBaseUrl" />
            </div>
            <p v-if="connectionStatus === 'ok'" class="text-sm text-green-600 mt-1">Connection successful</p>
            <p v-else-if="connectionStatus === 'fail'" class="text-sm text-red-600 mt-1">Connection failed</p>
            <p v-else class="text-sm text-slate-400 mt-1">
              Configure the API server URL if it's different from the web server
            </p>
          </FormItem>

          <FormItem for="kbAutosave" label="Auto-save to Knowledge Bank" layout="row">
            <ToggleSwitch v-model="config.data.kbAutosave" inputId="kbAutosave" @change="config.persist()" />
          </FormItem>

          <FormItem for="autoScroll" label="Auto-scroll to Bottom" layout="row">
            <ToggleSwitch v-model="settings.autoScroll" inputId="autoScroll" />
          </FormItem>
        </div>
      </TabPanel>
    </TabPanels>
  </Tabs>
</template>
