<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { RECOMMENDED_MODEL_PRESETS, type ModelPreset } from "../composables";

defineEmits<{ (e: "pick", value: ModelPreset): void }>();
</script>

<template>
  <div class="flex flex-wrap items-center gap-2">
    <span class="text-sm text-slate-500 dark:text-slate-400">Quick pick:</span>
    <span class="or-logo" role="img" aria-label="OpenRouter"></span>
    <button
      v-for="preset in RECOMMENDED_MODEL_PRESETS"
      :key="preset.model"
      type="button"
      :title="`${preset.provider} · ${preset.model}`"
      class="cursor-pointer rounded-full border border-slate-300 px-3 py-1 text-sm text-slate-700 transition-colors hover:border-slate-400 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-200 dark:hover:border-slate-500 dark:hover:bg-slate-800"
      @click="$emit('pick', preset)"
    >
      {{ preset.label }}
    </button>
  </div>
</template>

<style scoped>
/*
 * OpenRouter v2 logo as a CSS background image, swapped per colour scheme so
 * each theme keeps its branded variant (light: purple mark + dark wordmark;
 * dark: lime mark + light wordmark). The upstream assets are fixed-ink only.
 * Source: OpenRouterTeam/docs assets/logo-v2-{light,dark}.svg.
 */
.or-logo {
  display: inline-block;
  height: 18px;
  aspect-ratio: 1607.7629 / 293.71975;
  background-image: url("../assets/openrouter-logo-v2-light.svg");
  background-repeat: no-repeat;
  background-position: center;
  background-size: contain;
}

:global(.dark) .or-logo {
  background-image: url("../assets/openrouter-logo-v2-dark.svg");
}
</style>
