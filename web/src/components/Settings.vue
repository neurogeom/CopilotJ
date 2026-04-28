<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconRefresh, IconUpload } from "@tabler/icons-vue";
import { ref } from "vue";
import type { ThreadConfigModel, ThreadConfigQuery } from "../apis/threads";
import { useSettings } from "../store";

const emit = defineEmits<{
  (e: "submit", model: ThreadConfigQuery | null): void;
}>();

const settings = useSettings();

const fileInput = ref<HTMLInputElement | null>(null);

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
        <SettingModel :model="settings.model" :server-model="settings.serverModel" @update:model="sumbmitModel" />
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
