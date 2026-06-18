<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconArrowDown, IconLayoutSidebar, IconLayoutSidebarLeftExpand } from "@tabler/icons-vue";
import { nextTick, ref, useTemplateRef } from "vue";
import Dialog from "primevue/dialog";
import Button from "primevue/button";
import { useCustomScroll } from "../composables";
import { useActiveThread, useSettings } from "../store";
import { getModelName } from "../util";
import ChatboxHello from "./ChatboxHello.vue";
import ChatboxReactPosts from "./ChatboxReactPosts.vue";
import ChatInput from "./ChatInput.vue";
import FloatColorModeButton from "./FloatColorModeButton.vue";
import FloatThreadConfigButton from "./FloatThreadConfigButton.vue";

const emit = defineEmits<{
  (event: "toggleSidebar"): void;
}>();

const settings = useSettings();
const active = useActiveThread();

const postContainer = useTemplateRef("post-container");
const posts = useTemplateRef("posts");
const input = useTemplateRef("input");

const prompt = ref("");

// Dialog state for prompt optimization
const showOptimizeDialog = ref(false);
const originalPrompt = ref("");
const optimizedPrompt = ref("");

// Watch for posts changes to auto-scroll
active.thread.onUpdate(autoScroll);

const { arrivedBottom, scrollToBottom, scrollTo } = useCustomScroll(postContainer, { offsetBottom: 128 });

defineExpose({
  scrollToPost,
  reset,
});

function usePromptSuggestion(suggestion: string) {
  if (!active.thread.sendable) return;

  prompt.value = suggestion;
}

async function optimizePrompt(v: string) {
  if (!active.thread.sendable || !v) return;

  originalPrompt.value = v;
  const optimized = await active.thread.optimizePrompt(originalPrompt.value);
  if (optimized) {
    optimizedPrompt.value = optimized;
    showOptimizeDialog.value = true;
  }
}

function acceptOptimizedPrompt() {
  prompt.value = optimizedPrompt.value;
  showOptimizeDialog.value = false;
}

function rejectOptimizedPrompt() {
  showOptimizeDialog.value = false;
}

async function sendMessage() {
  if (!active.thread.sendable || !prompt.value) return;

  const value = prompt.value;
  input.value?.reset();
  await active.thread.send(value);
}

function autoScroll(force: boolean = false) {
  // check if currently at bottom
  if (force || (settings.autoScroll && arrivedBottom.value)) {
    nextTick(() => {
      scrollToBottom();
    });
  }
}

function scrollToPost(id: string) {
  const element = posts.value?.getElement(id);
  if (!element) {
    return;
  }

  // compute offset from top of container
  let offsetY = 0;
  let el: HTMLElement | null = element;
  const container = postContainer.value;
  while (el && el !== container) {
    offsetY += el.offsetTop;
    el = el.offsetParent as HTMLElement | null;
  }

  if (el !== container) {
    return;
  }

  scrollTo(offsetY - 50);
}

// TODO: move to store
function reset() {
  active.reset();
  active.thread.onUpdate(autoScroll);
  autoScroll(true);
  input.value?.reset();
}
</script>

<template>
  <main class="chatbox relative w-full h-full flex-1">
    <!-- Header -->
    <div class="absolute top-0 z-10 w-full p-4 pt-3 flex justify-between items-center backdrop-blur-xs">
      <div class="flex items-center gap-4">
        <div class="p-2 rounded-xl cursor-pointer" :class="{ 'bg-white/50 shadow': settings.expandSidebar }">
          <component
            :is="settings.expandSidebar ? IconLayoutSidebar : IconLayoutSidebarLeftExpand"
            class="hover:text-violet-500 transition-colors"
            @click="settings.toggleExpandSidebar()"
          />
        </div>

        <h1 class="text-xl font-bold">CopilotJ</h1>

        <template v-if="active.thread.status === 'started' && active.thread.config">
          <h2 class="text-slate-600">-</h2>

          <h2 class="text-slate-600">{{ getModelName(active.thread.config.model.name) }}</h2>
        </template>
      </div>

      <div class="mr-2 flex items-center gap-1">
        <FloatThreadConfigButton v-if="active.thread.status === 'started'" :thread="active.thread" />

        <FloatColorModeButton />
      </div>
    </div>

    <!-- Main Area -->
    <div class="flex-1 h-full flex flex-col">
      <div class="w-full h-full min-h-dvh flex flex-col text-slate-600 dark:text-slate-300">
        <div ref="post-container" class="relative h-full flex-1 overflow-y-auto">
          <!-- Chat Area -->
          <TransitionGroup name="message" tag="div" class="chatbox_chat w-full max-w-4xl mx-auto pt-20 px-4 space-y-4">
            <ChatboxReactPosts
              v-if="active.thread.posts.length > 0"
              ref="posts"
              :posts="active.thread.posts"
              :retrying="active.thread.retrying"
            />

            <ChatboxHello v-else @use-prompt-suggestion="usePromptSuggestion" />
          </TransitionGroup>

          <!-- Input Area -->
          <div class="chatbox_input sticky bottom-0 w-full flex flex-col justify-end">
            <div
              v-if="!arrivedBottom"
              class="chatbox_jump-btn-area absolute top-0 w-full flex justify-center items-center"
            >
              <div
                :class="[
                  'chatbox_jump-btn h-12 w-12 border-2 rounded-full flex justify-center items-center cursor-pointer',
                  'border-violet-500 bg-white/30 hover:bg-white/80 dark:bg-violet-500/30 dark:hover:bg-violet-500/50 transition-colors',
                ]"
                @click="scrollToBottom()"
              >
                <IconArrowDown class="text-violet-500" />
              </div>
            </div>

            <div class="w-full backdrop-blur-md mt-4">
              <ChatInput
                v-model="prompt"
                ref="input"
                class="w-full max-w-4xl mx-auto px-4 py-2"
                :sendable="active.thread.sendable"
                :loading="active.thread.loading"
                :optimizing="active.thread.optimizing"
                @send="sendMessage"
                @optimize-prompt="optimizePrompt"
                @reset="reset"
                @stop="active.thread.abort"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Prompt Optimization Dialog -->
    <Dialog v-model:visible="showOptimizeDialog" modal header="Optimized Prompt">
      <div class="space-y-4">
        <div>
          <p class="text-sm text-gray-500 mb-1">Original:</p>
          <p class="p-3 bg-gray-100 dark:bg-gray-800 rounded">{{ originalPrompt }}</p>
        </div>
        <div>
          <p class="text-sm text-gray-500 mb-1">Optimized:</p>
          <p class="p-3 bg-violet-50 dark:bg-violet-900/30 rounded">{{ optimizedPrompt }}</p>
        </div>
      </div>
      <template #footer>
        <Button label="Discard" severity="secondary" @click="rejectOptimizedPrompt" />
        <Button label="Accept" @click="acceptOptimizedPrompt" />
      </template>
    </Dialog>
  </main>
</template>

<style lang="postcss">
.chatbox {
  --bottom-min-height: 8rem;

  .chatbox_chat {
    min-height: calc(100vh - var(--bottom-min-height));
  }

  .chatbox_input {
    min-height: var(--bottom-min-height);
  }

  .chatbox_jump-btn-area {
    bottom: var(--bottom-min-height);
  }

  .chatbox_jump-btn {
    --bs-size: 25px;

    box-shadow: 0px 0px var(--bs-size) color-mix(in oklch, var(--color-violet-500) 50%, transparent);

    &:hover {
      --bs-size: 35px;
    }
  }
}
</style>
