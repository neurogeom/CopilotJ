<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import type { ManualTab } from "../assets/manual.md";

defineProps<{ tabs: ManualTab[] }>();
</script>

<template>
  <Tabs :value="tabs[0]?.id">
    <TabList>
      <Tab v-for="t in tabs" :key="t.id" :value="t.id">{{ t.name }}</Tab>
    </TabList>
    <TabPanels>
      <TabPanel v-for="t in tabs" :key="t.id" :value="t.id">
        <div v-html="t.html" />
      </TabPanel>
    </TabPanels>
  </Tabs>
</template>
