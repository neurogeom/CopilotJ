<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    for: string;
    label: string;
    layout?: "row" | "col";
    required?: boolean;
  }>(),
  { layout: "col", required: false },
);
</script>

<template>
  <div class="flex justify-between gap-2" :class="props.layout === 'col' ? 'flex-col' : 'flex-row items-center'">
    <label :for="props.for" class="text-sm font-medium text-slate-700 cursor-pointer"
      >{{ props.label }}<span v-if="props.required" class="text-red-500 ml-0.5">*</span></label
    >

    <slot />
  </div>
</template>
