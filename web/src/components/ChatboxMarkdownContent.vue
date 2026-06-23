<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import "highlight.js/styles/github-dark.css";

import { computed } from "vue";
import { createMarkdownRenderer } from "../lib";

const props = withDefaults(
  defineProps<{
    value: string;
    size?: "sm" | "md" | "lg";
  }>(),
  { size: "md" },
);

const RE_AT = /^@((\w*)|"[\w ]+") /;
const markdown = createMarkdownRenderer();

const hasAt = computed(() => RE_AT.test(props.value));

const atPrefix = computed(() => {
  const match = props.value.match(RE_AT);
  if (!match) return null;

  // Remove quotes if they exist
  let prefix = match[1];
  if (prefix.startsWith('"') && prefix.endsWith('"')) {
    prefix = prefix.slice(1, -1);
  }
  return prefix;
});

const valueWithoutAt = computed(() => {
  if (hasAt.value) {
    return props.value.replace(RE_AT, "");
  }
  return props.value;
});

const output = computed(() => {
  const html = markdown.parse(valueWithoutAt.value) as string;
  return html.replace(/^<p>(.*?)<\/p>/, "$1");
});
</script>

<template>
  <article
    class="prose dark:prose-invert w-full max-w-full prose-violet"
    :class="{
      'prose-sm': size === 'sm',
      'prose-lg': size === 'lg',
    }"
  >
    <span
      v-if="hasAt"
      class="inline-block px-3 py-1 mr-2 rounded-full bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-sm font-medium"
      >@{{ atPrefix }}</span
    >
    <div v-html="output" class="inline" />
  </article>
</template>

<style>
.prose pre code.hljs {
  background: transparent;
  padding: 0;
}

.prose :not(pre) > code.hljs,
.prose :not(pre) > code {
  border-radius: 0.375rem;
  padding: 0.15rem 0.35rem;
}
</style>
