<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script lang="ts" setup>
import { IconCirclePlus2, IconLoader2, IconPlayerStopFilled, IconSend, IconWand } from "@tabler/icons-vue";
import { useVModel } from "@vueuse/core";
import { useConfirm } from "primevue/useconfirm";
import { useTemplateRef, watch } from "vue";
import { useConfig, useSystemState } from "../store";

const props = defineProps<{
  modelValue: string;
  sendable: boolean;
  loading: boolean;
  optimizing: boolean;
}>();

const emit = defineEmits<{
  (event: "update:modelValue", value: string): void;
  (event: "send", value: string): void;
  (event: "reset"): void;
  (event: "optimizePrompt", value: string): void;
  (event: "stop"): void;
}>();

const textareaRef = useTemplateRef("textarea");

const modelValue = useVModel(props, "modelValue", emit);

const confirm = useConfirm();

const config = useConfig();
const state = useSystemState();

watch(
  () => props.loading,
  (newVal) => {
    if (!newVal) {
      confirm.close();
    }
  },
);

defineExpose({
  reset,
});

function handleKeyDown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send();
  }
}

function optimizePrompt() {
  if (!props.sendable || !modelValue.value) return;

  emit("optimizePrompt", modelValue.value);
}

function send() {
  if (!config.data.userAgreement || !props.sendable || !modelValue.value) return;

  emit("send", modelValue.value);
}

function stop(event: Event) {
  confirm.require({
    target: event.currentTarget as HTMLElement,
    message: "Are you sure you want to stop the conversation? This action cannot be undone.",
    rejectProps: {
      label: "Cancel",
      severity: "secondary",
      outlined: true,
    },
    acceptProps: {
      label: "Stop",
      severity: "danger",
    },
    accept() {
      emit("stop");
    },
  });
}

function adjustTextareaHeight(event: Event) {
  const textarea = event.target as HTMLTextAreaElement;
  textarea.style.height = "auto";

  const minHeight = parseInt(getComputedStyle(textarea).lineHeight) || 30;

  const newHeight = Math.max(textarea.scrollHeight, minHeight);
  textarea.style.height = `${newHeight}px`;
}

function reset() {
  modelValue.value = ""; // Clear input

  // Reset textarea height after clearing
  const textarea = textareaRef.value as HTMLTextAreaElement;
  if (textarea) {
    textarea.style.height = "auto";
    const minHeight = parseInt(getComputedStyle(textarea).lineHeight) || 30;
    textarea.style.height = `${minHeight}px`;
  }
}
</script>

<template>
  <div class="w-full max-w-4xl mx-auto px-4 py-2">
    <!-- Read-only banner until the User Agreement is accepted -->
    <div
      v-if="!config.data.userAgreement"
      class="flex items-center justify-between gap-3 rounded-lg border border-amber-400 bg-amber-50 px-4 py-2 text-amber-700 dark:border-amber-600 dark:bg-amber-950/40 dark:text-amber-300"
    >
      <p class="text-sm">Accept the <strong>User Agreement</strong> before sending messages.</p>
      <Button label="Open Settings" size="small" severity="secondary" @click="state.showSettings = true" />
    </div>

    <template v-else>
      <template v-if="props.sendable || props.loading || props.optimizing">
        <div class="flex gap-2 mb-1">
          <div
            :class="[
              'flex-1 flex rounded-lg px-2 py-1 border transition-colors  border-gray-500',
              props.sendable ? 'hover:border-violet-500 focus-within:border-violet-500' : '',
            ]"
          >
            <textarea
              ref="textarea"
              v-model="modelValue"
              :class="[
                'flex-1 px-2 py-1 focus:outline-none resize-none',
                props.sendable ? 'opacity-100' : 'opacity-50 cursor-not-allowed',
              ]"
              placeholder="Ask the agent to do anything related to ImageJ or image processing..."
              rows="1"
              :disabled="!props.sendable"
              @keydown="handleKeyDown"
              @input="adjustTextareaHeight"
            />

            <div class="flex items-center gap-2">
              <!-- Show loading while optimizing -->
              <IconLoader2 v-if="props.optimizing" size="20" class="animate-spin text-violet-500" />
              <!-- Show wand button in normal state -->
              <component
                v-else
                :is="IconWand"
                :class="[
                  'text-slate-500 transition-colors',
                  props.sendable ? 'hover:text-violet-500 cursor-pointer' : 'cursor-not-allowed',
                ]"
                @click="optimizePrompt"
              />
            </div>
          </div>

          <!-- Send button (shown when not loading) -->
          <div
            v-if="!props.loading"
            :class="[
              'px-4 py-2 rounded-lg flex items-center text-white transition-colors',
              props.sendable
                ? 'cursor-pointer bg-violet-500 hover:bg-violet-600 dark:hover:bg-violet-400'
                : 'cursor-not-allowed bg-violet-300 dark:bg-violet-700',
            ]"
            @click="send"
          >
            <IconSend size="28" />
          </div>

          <!-- Loading state with abort button -->
          <div v-else class="flex gap-2">
            <!-- Abort button -->
            <div
              class="relative px-4 py-2 rounded-lg flex items-center text-white bg-orange-400 hover:bg-orange-500 dark:hover:bg-orange-300 transition cursor-pointer"
              title="Stop conversation"
              @click="stop($event)"
            >
              <IconLoader2 size="28" class="animate-spin" />

              <IconPlayerStopFilled size="12" class="absolute top-1/2 left-1/2 translate-[-50%]" />
            </div>
          </div>
        </div>
      </template>

      <div v-else>
        <div
          :class="[
            'px-4 py-2 border rounded-lg flex justify-center items-center cursor-pointer transition-colors',
            'border-violet-500 text-violet-500 hover:text-white hover:bg-violet-500',
          ]"
          @click="emit('reset')"
        >
          <IconCirclePlus2 size="24" />

          <p class="mx-4">New Thread</p>
        </div>
      </div>
    </template>
  </div>
</template>
