<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import {
  IconChevronDown,
  IconChevronUp,
  IconCopy,
  IconExclamationCircleFilled,
  IconLoader2,
  IconRefresh,
  IconSquareRoundedCheckFilled,
  IconTools,
  IconUserFilled,
} from "@tabler/icons-vue";
import { useTimestamp } from "@vueuse/core";
import type { Dayjs } from "dayjs";
import dayjs from "dayjs";
import { computed, ref, watch } from "vue";
import type { Post, RetryingState } from "../store";
import ChatboxMarkdownContent from "./ChatboxMarkdownContent.vue";

const props = defineProps<{
  posts: Post[];
  retrying: RetryingState | null;
}>();

const itemRefs = ref<Record<string, HTMLElement>>({});

const hiddenReasoning = ref(new Set<number>());
const showToolCallDetails = ref(new Set<string>());
const copyStatus = ref<Record<number, boolean>>({}); // Track copy status for each post

const timestamp = useTimestamp();

// Live countdown for the 429 retry bar: reuse the reactive ``timestamp`` (updates
// ~each animation frame) so "waiting Ns" ticks down instead of freezing on a
// static backoff value. The start point resets on every new update:retry event.
const retryStartedAt = ref(0);
const remainingSeconds = computed(() => {
  if (!props.retrying) return 0;
  const elapsed = (timestamp.value - retryStartedAt.value) / 1000;
  return Math.max(0, Math.ceil(props.retrying.wait - elapsed));
});
watch(
  () => props.retrying,
  (r) => {
    if (r) retryStartedAt.value = Date.now();
  },
);

defineExpose({
  getElement,
});

function copyToClipboard(text: string, postIndex: number) {
  navigator.clipboard
    .writeText(text)
    .then(() => {
      // Set copy status to true (success)
      copyStatus.value[postIndex] = true;
      // Reset status after 2 seconds
      setTimeout(() => {
        copyStatus.value[postIndex] = false;
      }, 2000);
    })
    .catch((err) => {
      console.error("Failed to copy: ", err);
      // Set copy status to false (error)
      copyStatus.value[postIndex] = false;
    });
}

function shouldShowHeader(message: Post, index: number): boolean {
  if (index === 0 || isPost(message)) {
    return true;
  }

  const previousItem = props.posts[index - 1];
  if (!isPost(previousItem)) {
    return true;
  }

  switch (message.type) {
    case "user":
      return previousItem.type !== "user";

    case "agent":
      return previousItem.type !== "agent" || message.role !== previousItem.role;

    default:
      return false;
  }
}

function toggleReasoning(index: number) {
  if (hiddenReasoning.value.has(index)) {
    hiddenReasoning.value.delete(index);
  } else {
    hiddenReasoning.value.add(index);
  }
}

function toggleToolResult(key: string) {
  if (showToolCallDetails.value.has(key)) {
    showToolCallDetails.value.delete(key);
  } else {
    showToolCallDetails.value.add(key);
  }
}

function isPost(message: Post): boolean {
  return !message.type.startsWith("error:") && !message.type.startsWith("system:");
}

function getAgentName(post: Post): string {
  switch (post.type) {
    case "agent":
      const role = post.role.startsWith("agent:") ? post.role.replace("agent:", "") : post.role;
      return role.charAt(0).toUpperCase() + role.slice(1).replace(/_/g, " ");

    case "user":
      return "You";

    case "error:system":
      return "System Error";

    case "error:assistant":
      return "Assistant Error";

    default:
      return "Unknown";
  }
}

function formatTime(t: Dayjs): string {
  return t.format("HH:mm");
}

function formatDiffSeconds(start: Dayjs, end?: Dayjs): string {
  if (!end) {
    end = dayjs(timestamp.value);
  }

  const sec = end.diff(start, "millisecond") / 1000;
  const precision = sec < 10 ? 1 : 0;
  return sec.toFixed(precision) + "s";
}

function getElement(id: string): HTMLElement | null {
  return itemRefs.value[id] ?? null;
}
</script>

<template>
  <div class="chatbox-react-posts flex flex-col w-full text-black dark:text-slate-100">
    <template v-for="(post, index) in props.posts" :key="index">
      <div
        v-if="isPost(post)"
        :ref="(el) => (itemRefs[post.id] = el as HTMLElement)"
        class="flex flex-col w-full gap-1"
      >
        <div v-if="shouldShowHeader(post, index)" class="mt-2 flex items-center">
          <div
            v-if="post.type === 'agent'"
            class="w-8 h-8 rounded-full flex items-center justify-center"
            :class="{
              'bg-blue-500 dark:bg-blue-600': getAgentName(post) == 'Leader Agent',
              'bg-emerald-500 dark:bg-emerald-600': getAgentName(post) == 'Research Agent',
              'bg-amber-500 dark:bg-amber-600': getAgentName(post) == 'Tool Agent',
            }"
          >
            <img class="w-5 h-5 object-contain" src="../assets/copilotj.png" />
          </div>

          <div
            v-else-if="post.type === 'user'"
            class="w-8 h-8 rounded-full flex items-center justify-center overflow-hidden bg-violet-500"
          >
            <IconUserFilled class="w-6 h-6 rounded-full text-white" />
          </div>

          <span class="mx-2 pl-1 font-medium">
            {{ post.type === "user" ? "You" : getAgentName(post) }}
          </span>
        </div>

        <div class="flex-1 rounded-lg flex flex-col transition-all duration-300">
          <!-- Thought -->
          <div v-if="post.type === 'agent' && post.thought.length > 0" class="text-slate-600 dark:text-slate-300">
            <div class="chatbox-react-posts_message-box flex">
              <div class="w-10 flex">
                <span class="chatbox-react-posts_message-hover-in mt-3 text-xs transition-all duration-75">
                  {{ formatTime(post.createAt) }}
                </span>
              </div>

              <div
                :class="[
                  'mt-1 p-2 rounded-lg flex-1 flex justify-between items-center gap-2 cursor-pointer transition hover:shadow',
                  'hover:bg-white hover:dark:bg-gray-800/80',
                ]"
                @click="toggleReasoning(index)"
              >
                <div class="text-xs">
                  <span class="font-bold">Thought</span>
                  <span class="ml-2">take {{ formatDiffSeconds(post.createAt, post.content?.createAt) }}</span>
                </div>

                <component :is="hiddenReasoning.has(index) ? IconChevronUp : IconChevronDown" size="16" />
              </div>
            </div>

            <div v-if="!hiddenReasoning.has(index)" class="pt-0 pb-2 rounded-lg flex flex-col gap-2 text-sm">
              <template v-for="(item, itemIndex) in post.thought" :key="itemIndex">
                <div class="chatbox-react-posts_message-box flex">
                  <div class="w-10 flex">
                    <span class="chatbox-react-posts_message-hover-in mt-1 text-xs transition-all duration-75">
                      {{ formatTime(item.createAt) }}
                    </span>
                  </div>

                  <!-- Reasoning -->
                  <ChatboxMarkdownContent
                    v-if="item.type === 'reasoning'"
                    :value="item.content"
                    size="sm"
                    class="ml-4"
                  />

                  <!-- Action -->
                  <div
                    v-else-if="item.type === 'action'"
                    :class="[
                      'min-w-0  flex-1 ml-4 rounded-lg transition-all flex flex-row',
                      showToolCallDetails.has(`${index}-${itemIndex}`) ? ' bg-white/50 dark:bg-white/10' : '',
                    ]"
                  >
                    <div
                      :class="[
                        'w-0 min-h-full border-l-6 border-r-6 rounded-l-lg transition-all cursor-pointer',
                        showToolCallDetails.has(`${index}-${itemIndex}`)
                          ? 'border-gray-300 dark:border-gray-800'
                          : 'border-r-gray-500 dark:border-r-gray-400 border-l-transparent',
                      ]"
                      @click="toggleToolResult(`${index}-${itemIndex}`)"
                    />

                    <div
                      :class="[
                        'min-w-0 flex-1 rounded-r-lg  transition',
                        showToolCallDetails.has(`${index}-${itemIndex}`) ? 'shadow hover:shadow-md' : 'hover:shadow',
                      ]"
                    >
                      <div
                        :class="[
                          'p-2 rounded-r-lg flex flex-col gap-1 cursor-pointer',
                          'hover:bg-white/50 dark:hover:bg-white/10',
                        ]"
                        @click="toggleToolResult(`${index}-${itemIndex}`)"
                      >
                        <div class="w-full flex items-center gap-2">
                          <template v-if="item.result">
                            <IconSquareRoundedCheckFilled
                              v-if="item.result.type === 'success'"
                              class="text-emerald-500"
                            />

                            <IconExclamationCircleFilled v-if="item.result.type === 'error'" class="text-yellow-500" />
                          </template>

                          <IconLoader2 v-else-if="item.called" class="text-blue-500 animate-spin" />

                          <IconTools v-else />

                          <h4 class="font-bold">{{ item.content.tool.display_name }}</h4>

                          <div
                            class="min-w-0 flex-1 text-nowrap text-ellipsis overflow-hidden"
                            :title="item.content.tool.description"
                          >
                            {{ item.content.tool.description }}
                          </div>
                        </div>
                      </div>

                      <div v-if="showToolCallDetails.has(`${index}-${itemIndex}`)" class="p-3 pt-1 flex flex-col gap-2">
                        <div v-if="Object.keys(item.content.args).length">
                          <h5 class="text-xs text-slate-400">Arguments:</h5>

                          <p v-for="(v, k) in item.content.args" :key="k">
                            <strong>{{ k }}</strong
                            ><span>: {{ v }}</span>
                          </p>
                        </div>

                        <div v-if="item.result">
                          <h5 class="text-xs text-slate-400">
                            {{ item.result.type === "success" ? "Result" : "Error" }}:
                          </h5>

                          <ChatboxMarkdownContent
                            class="chatbox-react-posts_action-item"
                            size="sm"
                            :value="item.result.data"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </div>

          <!-- Content -->
          <div v-if="post.content" class="chatbox-react-posts_message-box flex mt-1">
            <div class="w-10 flex">
              <span class="chatbox-react-posts_message-hover-in mt-2 text-xs transition-all duration-75">
                {{ formatTime(post.content.createAt) }}
              </span>
            </div>

            <div class="w-full flex-1 flex flex-col">
              <div
                :class="[
                  'chatbox-react-posts_message-hover-shadow-md rounded-lg p-4 pl-5 flex transition-all duration-300 shadow',
                  post.type === 'user' ? 'bg-violet-100  dark:bg-violet-950/80' : 'bg-white dark:bg-gray-800/80',
                  { 'animate-pulse': post.type == 'agent' && post.streaming },
                ]"
              >
                <ChatboxMarkdownContent :value="post.content.data" />
              </div>

              <!-- Copy Button -->
              <div class="chatbox-react-posts_message-hover-in flex justify-start mt-1 ml-2">
                <button
                  @click="copyToClipboard(post.content.data, index)"
                  :class="[
                    'flex items-center gap-1 text-xs transition',
                    'text-slate-700 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200',
                  ]"
                  title="Copy to clipboard"
                >
                  <IconCopy size="14" />

                  <span>{{ copyStatus[index] ? "Copied!" : "Copy" }}</span>
                </button>
              </div>
            </div>
          </div>

          <!-- Thinking animation -->
          <div
            v-if="index === props.posts.length - 1 && post.type === 'agent' && post.streaming"
            class="ml-12 flex items-center gap-2 text-sm text-slate-700 dark:text-slate-500 animate-pulse"
          >
            <p v-if="post.content">🤗 Answering...</p>
            <p v-else>🤔 Thinking...</p>
          </div>

          <!-- 429 auto-retry feedback (#96) -->
          <div
            v-if="props.retrying && index === props.posts.length - 1"
            class="ml-12 flex items-center gap-2 text-sm text-amber-600 dark:text-amber-400 animate-pulse"
          >
            <IconRefresh size="16" class="animate-spin" />
            <span>
              Rate limited — retrying ({{ props.retrying.attempt }}/{{ props.retrying.maxAttempts }}, waiting
              {{ remainingSeconds }}s)…
            </span>
          </div>
        </div>
      </div>

      <!-- Dialog -->
      <div
        v-else-if="post.type === 'system:dialog_completed'"
        :ref="(el) => (itemRefs[post.id] = el as HTMLElement)"
        class="chatbox-react-posts_message-box flex my-1"
      >
        <div class="w-10 flex">
          <span class="chatbox-react-posts_message-hover-in text-xs transition-all duration-75">
            {{ post.content && formatTime(post.content?.createAt) }}
          </span>
        </div>

        <div class="flex-1 mx-4 flex items-center gap-4">
          <div class="flex-1 border-b border-gray-400 dark:border-gray-600" />

          <p class="text-center text-sm text-gray-400 dark:text-gray-600">Dialog {{ post.content?.data }} completed</p>

          <div class="flex-1 border-b border-gray-400 dark:border-gray-600" />
        </div>
      </div>

      <!-- System -->
      <div
        v-else
        :ref="(el) => (itemRefs[post.id] = el as HTMLElement)"
        class="chatbox-react-posts_message-box flex my-4"
      >
        <div class="w-10 flex">
          <span class="chatbox-react-posts_message-hover-in mt-2 text-xs transition-all duration-75">
            {{ post.content && formatTime(post.content?.createAt) }}
          </span>
        </div>

        <div
          :class="[
            'flex-1 mx-16 rounded-md border p-2 text-sm transition-colors',
            'text-red-800 dark:text-red-400 border-red-200 dark:border-red-500/30 bg-red-50 dark:bg-red-900/20',
            'hover:border-red-300 dark:hover:border-red-500/50',
          ]"
        >
          <p class="text-center">{{ post.content?.data }}</p>
        </div>
      </div>
    </template>
  </div>
</template>

<style lang="postcss">
@reference "../style.css";

.chatbox-react-posts .chatbox-react-posts_message-box {
  .chatbox-react-posts_message-hover-in {
    opacity: 0;
  }

  &:hover .chatbox-react-posts_message-hover-in {
    opacity: 0.5;
  }

  &:hover .chatbox-react-posts_message-hover-shadow-md {
    @apply shadow-md;
  }
}
</style>
