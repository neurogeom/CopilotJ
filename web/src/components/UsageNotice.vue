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
        CopilotJ uses large language models (LLMs) to plan and execute your image-analysis workflows. Before you
        proceed, please read and acknowledge the following:
      </p>
      <ol class="list-decimal list-inside space-y-2 pl-2">
        <li>
          <strong>Text data is sent to model providers.</strong> Your task descriptions and processing instructions are
          transmitted to an LLM. When the model is hosted by an external provider (e.g. OpenAI, Anthropic, Google, or
          another third-party service), these requests are sent over the internet to that provider.
        </li>
        <li>
          <strong>Vision data is opt-in.</strong> When Vision is enabled, CopilotJ may additionally transmit image
          snapshots (e.g. screenshots of the ImageJ interface — including displayed image content and incidental
          on-screen metadata such as file paths, window titles, or identifiers) to the model. Vision is
          <strong>disabled by default</strong>.
        </li>
        <li>
          <strong>Third-party handling is beyond CopilotJ's control.</strong> Confidentiality is governed solely by the
          provider's data-handling and retention policies.
        </li>
        <li>
          <strong>You can reduce the risk.</strong> Keep Vision disabled, or configure a
          <strong>locally deployed model</strong> to avoid sending sensitive data to any third party.
        </li>
      </ol>
      <p class="text-slate-500 dark:text-slate-400">
        If your work involves confidential, proprietary, or personally identifiable data, consult your organization's
        data-governance policy before using online model providers.
      </p>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox v-model="agreed" binary inputId="userAgreement" />
      <label for="userAgreement" class="cursor-pointer select-none text-sm">
        I have read and understood the above and wish to use CopilotJ.
      </label>
    </div>

    <div class="flex flex-col gap-1">
      <div class="flex items-center gap-2">
        <Checkbox v-model="enableVision" binary inputId="enableVision" />
        <label for="enableVision" class="cursor-pointer select-none text-sm">
          Enable Vision — send image snapshots to the model when image interpretation is needed.
        </label>
      </div>
      <p class="pl-6 text-xs text-slate-500 dark:text-slate-400">Recommended to enable.</p>
    </div>

    <hr class="border-slate-200 dark:border-slate-700" />
  </section>
</template>
