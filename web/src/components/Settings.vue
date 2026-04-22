<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconRefresh, IconUpload } from "@tabler/icons-vue";
import { computed, onMounted, ref } from "vue";
import type { ThreadConfigModel, ThreadConfigQuery, ClientInfo } from "../apis";
import { listClients } from "../apis";
import { useSettings } from "../store";
import SettingModel from "./SettingModel.vue";

const emit = defineEmits<{
  (e: "submit", model: ThreadConfigQuery | null): void;
}>();

const settings = useSettings();

const fileInput = ref<HTMLInputElement | null>(null);
const accessTokenInput = ref(settings.accessToken);
const clients = ref<ClientInfo[]>([]);

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

function submitAccessToken() {
  settings.setAccessToken(accessTokenInput.value);
}

async function refreshClients() {
  try {
    clients.value = await listClients();
  } catch {
    clients.value = [];
  }
}

const clientOptions = computed(() =>
  clients.value.map((c) => ({
    label: c.id.substring(0, 8) + "...",
    value: c.id,
  })),
);

function onClientSelected() {
  settings.setSelectedClientId(settings.selectedClientId);
}

onMounted(refreshClients);
</script>

<template>
  <Tabs value="model">
    <TabList>
      <Tab value="model">Model</Tab>
      <Tab value="bridge">Bridge</Tab>
      <Tab value="kb">Knowledge Base</Tab>
      <Tab value="pref">Preferences</Tab>
    </TabList>

    <TabPanels>
      <!-- Model Tab -->
      <TabPanel value="model">
        <SettingModel :model="settings.model" @update:model="sumbmitModel" />
      </TabPanel>

      <!-- Bridge Tab -->
      <TabPanel value="bridge">
        <div class="space-y-4">
          <FormItem for="bridgeSelect" label="Bridge">
            <div class="flex items-center gap-2">
              <Select
                id="bridgeSelect"
                v-model="settings.selectedClientId"
                :options="clientOptions"
                optionLabel="label"
                optionValue="value"
                placeholder="Select a bridge..."
                class="flex-1"
                @change="onClientSelected"
              />
              <Button size="small" severity="secondary" @click="refreshClients">
                <IconRefresh size="14" />
              </Button>
            </div>
          </FormItem>

          <FormItem for="accessToken" label="Access Token">
            <InputText
              id="accessToken"
              v-model="accessTokenInput"
              type="password"
              class="w-full"
              placeholder="Enter bridge access token"
              @change="submitAccessToken"
            />
          </FormItem>
          <p class="text-sm text-slate-500">
            Select a bridge above, then enter its access token to connect.
          </p>
        </div>
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
          <FormItem for="autoScroll" label="Auto-scroll to Bottom" layout="row">
            <ToggleSwitch v-model="settings.autoScroll" inputId="autoScroll" />
          </FormItem>
        </div>
      </TabPanel>
    </TabPanels>
  </Tabs>
</template>
