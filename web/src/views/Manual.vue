<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import "highlight.js/styles/github-dark.css";
import { nextTick, onMounted, shallowRef } from "vue";
import { useRoute } from "vue-router";
import { blocks as initialBlocks, toc as initialToc, type ManualBlock, type ManualTocItem } from "../assets/manual.md";
import ManualTabs from "../components/ManualTabs.vue";

type TocItem = ManualTocItem;
type Block = ManualBlock;

// The markdown is parsed, rendered (header IDs + syntax highlighting), `::: tabs`
// regions turned into tab blocks, and a TOC built — all at build time by the
// manual-markdown Vite plugin. Here we only consume the result.
const blocks = shallowRef<Block[]>(initialBlocks);
const toc = shallowRef<TocItem[]>(initialToc);
const route = useRoute();

// HMR: update content in place without remounting the component, so onMounted does
// not re-run and the scroll position is preserved (no jump on every edit).
if (import.meta.hot) {
  import.meta.hot.accept("../assets/manual.md", (mod) => {
    if (!mod) return;
    blocks.value = mod.blocks;
    toc.value = mod.toc;
  });
}

function scrollToSection(id: string) {
  const node = document.getElementById(id);
  if (!node) return;
  // Reveal the answer when jumping to a collapsible FAQ entry.
  if (node instanceof HTMLDetailsElement) node.open = true;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
}

// In-content markdown links such as [Provider quick reference](#provider-quick-reference)
// render as plain <a href="#...">. Under hash routing the browser would treat a bare
// "#id" as a route change (and "#/manual#id" only scrolls on a fresh mount), so intercept
// plain left-clicks on the rendered manual and scroll to the target in place.
function anchorIdFromHref(href: string | null): string | null {
  if (!href || !href.startsWith("#")) return null;
  // Route-aware form "#/<route>#<id>": take the segment after the second '#'.
  if (href.startsWith("#/")) {
    const second = href.indexOf("#", 2);
    return second === -1 ? null : decodeURIComponent(href.slice(second + 1));
  }
  return decodeURIComponent(href.slice(1));
}

function onContentClick(event: MouseEvent) {
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
    return;
  const anchor = (event.target as HTMLElement | null)?.closest("a");
  if (!anchor) return;
  const id = anchorIdFromHref(anchor.getAttribute("href"));
  // Only intercept when the anchor resolves to an element on this page; let
  // external links and unknown anchors fall through to the browser default.
  if (!id || !document.getElementById(id)) return;
  event.preventDefault();
  scrollToSection(id);
}

// Deep-link scroll: run exactly once on first mount (e.g. opening the plugin's FAQ
// link in a fresh tab). HMR edits do not remount, so this never fires repeatedly.
onMounted(async () => {
  // External deep links (e.g. the plugin's MCP-unavailable link) arrive as ?section=<id>,
  // because java.net.URI rejects a '#' inside the hash-routed fragment. In-page links still
  // arrive as route.hash.
  const section = route.query.section;
  const id = typeof section === "string" ? section : route.hash ? decodeURIComponent(route.hash.slice(1)) : null;
  if (!id) return;
  await nextTick();
  scrollToSection(id);
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
          <article
            class="prose dark:prose-invert prose-emerald max-w-none prose-headings:scroll-mt-28"
            @click="onContentClick"
          >
            <template v-for="(block, i) in blocks" :key="i">
              <div v-if="block.type === 'html'" v-html="block.html" />
              <ManualTabs v-else :tabs="block.tabs" class="my-6" />
            </template>
          </article>
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

/* Collapsible <details> (FAQ entries + the ImageJ Updater steps in manual.md). */
.prose details {
  margin: 1.25rem 0;
  border: 1px solid rgb(228 228 231 / 0.7);
  border-radius: 0.75rem;
  background: rgb(255 255 255 / 0.6);
  padding: 0.75rem 1rem;
}

.prose details > summary {
  cursor: pointer;
  font-weight: 600;
  color: rgb(39 39 42);
  list-style: none;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.prose details > summary::-webkit-details-marker {
  display: none;
}

.prose details > summary::before {
  content: "▸";
  color: rgb(161 161 170);
  transition: transform 0.15s ease;
  display: inline-block;
}

.prose details[open] > summary::before {
  transform: rotate(90deg);
}

/* PrimeVue's TabPanel sets its own text color, which breaks inheritance of the prose
   body color into tab panels. Restore it so tab content matches the rest of the manual. */
.prose .p-tabpanel {
  color: var(--tw-prose-body);
}
</style>
