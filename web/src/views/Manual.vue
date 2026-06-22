<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import "highlight.js/styles/github-dark.css";
import { nextTick, onMounted, shallowRef } from "vue";
import { useRoute } from "vue-router";
import { html as initialHtml, toc as initialToc, type ManualTocItem } from "../assets/manual.md";

type TocItem = ManualTocItem;

// The markdown is parsed, rendered (with header IDs + syntax highlighting), and
// turned into a TOC at build time by the manual-markdown Vite plugin. Here we only
// consume the result.
const html = shallowRef(initialHtml);
const toc = shallowRef<TocItem[]>(initialToc);
const route = useRoute();

// HMR: update the content in place without remounting the component, so onMounted
// does not re-run and the scroll position is preserved (no jump on every edit).
if (import.meta.hot) {
  import.meta.hot.accept("../assets/manual.md", (mod) => {
    if (!mod) return;
    html.value = mod.html;
    toc.value = mod.toc;
  });
}

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Deep-link scroll: run exactly once on first mount (e.g. opening the plugin's FAQ
// link in a fresh tab). HMR edits do not remount, so this never fires repeatedly.
onMounted(async () => {
  const hash = route.hash;
  if (!hash) return;
  await nextTick();
  scrollToSection(decodeURIComponent(hash.slice(1)));
});
</script>

<template>
  <section class="py-12 md:py-20">
    <div class="mx-auto w-full max-w-7xl px-6 md:px-10">
      <div class="max-w-3xl">
        <p class="text-sm font-semibold uppercase tracking-[0.24em] text-emerald-700">Documentation</p>

        <h2 class="mt-3 text-3xl font-bold tracking-tight md:text-4xl">User Manual</h2>
      </div>

      <div class="mt-8 grid grid-cols-1 gap-8 xl:grid-cols-[260px_minmax(0,1fr)]">
        <aside class="xl:sticky xl:top-24 xl:self-start">
          <div class="rounded-3xl border border-zinc-200/60 bg-white/80 p-5 shadow-sm ring-1 ring-black/5">
            <div class="text-xs font-semibold uppercase tracking-[0.22em] text-zinc-500">On this page</div>

            <ul v-if="toc.length" class="mt-4 space-y-1.5 text-sm">
              <li v-for="item in toc" :key="item.id">
                <button
                  type="button"
                  class="block rounded-xl px-3 py-2 transition-colors hover:bg-zinc-100 text-left"
                  :class="item.level === 3 ? 'ml-3 text-zinc-500' : 'font-medium text-zinc-800'"
                  @click="scrollToSection(item.id)"
                >
                  {{ item.text }}
                </button>
              </li>
            </ul>

            <p v-else class="mt-4 text-sm text-zinc-500">No sections detected</p>
          </div>
        </aside>

        <div class="rounded-4xl border border-zinc-200/60 bg-white/85 p-6 shadow-sm ring-1 ring-black/5 md:p-8 xl:p-10">
          <article class="prose dark:prose-invert prose-emerald max-w-none prose-headings:scroll-mt-28" v-html="html" />
        </div>
      </div>
    </div>
  </section>
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
