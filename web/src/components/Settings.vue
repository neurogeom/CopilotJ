<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconRefresh, IconUpload } from "@tabler/icons-vue";
import { ref } from "vue";
import type { ThreadConfigModel, ThreadConfigQuery } from "../apis";
import { getBaseUrl, isApiBaseConfigurable, setApiBaseUrl, testApiConnection } from "../apis/base";
import { useSettings } from "../store";
import SettingModel from "./SettingModel.vue";

const emit = defineEmits<{
  (e: "submit", model: ThreadConfigQuery | null): void;
}>();

const settings = useSettings();

const fileInput = ref<HTMLInputElement | null>(null);

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

function handleFileUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) {
    console.log("Files selected:", input.files);
  }
}

function sumbmitModel(model: ThreadConfigModel | null) {
  settings.setModel(model);
  emit("submit", { model: settings.model });
}
</script>

<template>
  <Tabs value="model">
    <TabList>
      <Tab value="model">Model</Tab>
      <Tab value="kb">Knowledge Base</Tab>
      <Tab value="pref">Preferences</Tab>
    </TabList>

    <TabPanels>
      <!-- Model Tab -->
      <TabPanel value="model">
        <SettingModel :model="settings.model" @update:model="sumbmitModel" />
      </TabPanel>

      <!-- Knowledge Base Tab -->
      <TabPanel value="kb">
        <div class="space-y-4">
          <FormItem for="knowledgeBaseEnabled" label="Knowledge Base">
            <div
              class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center"
              @dragover.prevent
              @drop.prevent="handleFileUpload"
            >
              <input type="file" ref="fileInput" class="hidden" multiple @change="handleFileUpload" />

              <div class="flex flex-col items-center">
                <IconUpload class="text-slate-400" size="42" />

                <p class="mt-4 text-slate-600">Click to upload or drag files here</p>
              </div>
            </div>
          </FormItem>

          <FormItem for="activeKnowledgeBase" label="Active Knowledge Base">
            <Select class="w-full" label-id="activeKnowledgeBase" />
          </FormItem>

          <Button class="w-full">
            <IconRefresh />
            Reindex Knowledge Base
          </Button>

          <div class="flex items-center justify-between">
            <span class="text-sm font-medium text-slate-700">Knowledge Base Status</span>

            <span class="rounded-full px-2 py-1 bg-green-100 text-green-800 text-sm">ready</span>
          </div>
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

          <FormItem for="autoScroll" label="Auto-scroll to Bottom" layout="row">
            <ToggleSwitch v-model="settings.autoScroll" inputId="autoScroll" />
          </FormItem>
        </div>
      </TabPanel>
    </TabPanels>
  </Tabs>
</template>
