<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed } from "vue";
import { IconAlertTriangle } from "@tabler/icons-vue";

const props = defineProps<{
  userAgreement: boolean;
  visionEnabled: boolean;
}>();
const emit = defineEmits<{
  (e: "update:userAgreement", value: boolean): void;
  (e: "update:visionEnabled", value: boolean): void;
}>();

// Controlled checkboxes: the parent owns the state and commits it (the Wizard on
// Finish, Settings on Save) — consistent with the other edit-then-save tabs.
const agreed = computed({
  get: () => props.userAgreement,
  set: (v: boolean) => emit("update:userAgreement", v),
});
const enableVision = computed({
  get: () => props.visionEnabled,
  set: (v: boolean) => emit("update:visionEnabled", v),
});
</script>

<template>
  <section class="flex flex-col gap-4">
    <div class="flex items-center gap-3 text-amber-500">
      <IconAlertTriangle size="24" />
      <h3 class="text-lg font-semibold">Privacy &amp; Data Handling Notice</h3>
    </div>

    <div class="space-y-3 text-sm leading-relaxed text-slate-600 dark:text-slate-300">
      <p>
        CopilotJ uses large language models (LLMs) to plan bioimage-analysis tasks and coordinate their execution within
        the ImageJ environment. Before proceeding, please review the following information:
      </p>
      <ol class="list-decimal list-inside space-y-2 pl-2">
        <li>
          <strong>Text data is sent to model providers.</strong> Your task descriptions and processing instructions are
          transmitted to the configured LLM. When the model is hosted by an external provider (e.g., OpenAI, Anthropic,
          Google, or other third-party providers), these requests are sent over the internet to the corresponding
          service.
        </li>
        <li>
          <strong>Visual data is transmitted to model providers on an opt-in basis.</strong> When Vision is enabled,
          CopilotJ may additionally transmit image snapshots (e.g. screenshots of the ImageJ interface, displayed image
          content, and on-screen metadata such as file paths, window titles, or identifiers) to the model. Vision is
          disabled by default.
        </li>
        <li>
          <strong>Data handling depends on the selected provider.</strong> Information transmitted to a cloud-hosted
          model is subject to the data-handling, security, and retention policies of the corresponding provider. Users
          should review and evaluate these policies before using external services.
        </li>
        <li>
          <strong>Local deployment can reduce privacy risks.</strong> Users can use locally deployed models to avoid
          transmitting sensitive data to third-party providers.
        </li>
      </ol>
      <p class="text-slate-500 dark:text-slate-400">
        If your work involves confidential, proprietary, protected, or personally identifiable information, consult your
        organization's data-governance policy before using cloud-hosted AI services.
      </p>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox v-model="agreed" binary inputId="userAgreement" />
      <label for="userAgreement" class="cursor-pointer select-none text-sm">
        I have read and understood the above information and agree to use CopilotJ under these conditions.
      </label>
    </div>

    <div class="flex flex-col gap-1">
      <div class="flex items-center gap-2">
        <Checkbox v-model="enableVision" binary inputId="enableVision" />
        <label for="enableVision" class="cursor-pointer select-none text-sm">
          I choose to enable Vision support and allow CopilotJ to send image snapshots to the configured model when
          image interpretation is required.
        </label>
      </div>
      <p class="pl-6 text-xs text-slate-500 dark:text-slate-400">Optional but recommended.</p>
    </div>

    <hr class="border-slate-200 dark:border-slate-700" />
  </section>
</template>
