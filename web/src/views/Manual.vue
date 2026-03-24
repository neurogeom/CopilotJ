<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import "highlight.js/styles/github-dark.css";
import { computed, nextTick, onMounted, ref } from "vue";
import { createMarkdownRenderer } from "../lib/markdown";

type TocItem = { level: number; text: string; id: string };

function slugify(input: string, counts: Map<string, number>) {
  let s = input
    .toLowerCase()
    .trim()
    .replace(/[`*_~]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-");
  if (!s) s = "section";
  const n = counts.get(s) || 0;
  counts.set(s, n + 1);
  return n ? `${s}-${n}` : s;
}

function getHeadingText(token: any): string {
  if (token.type === "text") return token.text;
  if (token.tokens) {
    return token.tokens.map(getHeadingText).join(" ");
  }
  return "";
}

const headingCounts = new Map<string, number>();

const renderer = {
  heading({ tokens, depth }: { tokens: any[]; depth: number }) {
    const text = tokens.map(getHeadingText).join(" ");
    const id = slugify(text, headingCounts);
    const tag = `h${depth}`;
    return `<${tag} id="${id}">${text}</${tag}>\n`;
  },
};

const markdown = createMarkdownRenderer(renderer);
const loadManualModule = () => import("../assets/manual.md?raw");

const manualSrc = ref("");
const isLoading = ref(true);
const loadError = ref("");
const pendingScrollY = ref<number | null>(import.meta.hot?.data.manualScrollY ?? null);

const toc = computed<TocItem[]>(() => {
  if (!manualSrc.value) return [];

  const tokens = markdown.lexer(manualSrc.value);
  const tocCounts = new Map<string, number>();
  const items: TocItem[] = [];

  for (const token of tokens) {
    if (token.type === "heading" && (token.depth === 2 || token.depth === 3)) {
      const text = token.tokens?.map(getHeadingText).join(" ") || "";
      const id = slugify(text, tocCounts);
      items.push({ level: token.depth, text, id });
    }
  }

  return items;
});

const html = computed(() => {
  if (!manualSrc.value) return "";
  headingCounts.clear();
  return markdown.parse(manualSrc.value) as string;
});

async function loadManual() {
  isLoading.value = true;
  loadError.value = "";

  try {
    const module = await loadManualModule();
    manualSrc.value = module.default;
  } catch (error) {
    loadError.value = error instanceof Error ? error.message : "Failed to load the manual.";
  } finally {
    isLoading.value = false;
    if (pendingScrollY.value != null) {
      const scrollY = pendingScrollY.value;
      pendingScrollY.value = null;
      await nextTick();
      requestAnimationFrame(() => {
        window.scrollTo({ top: scrollY, behavior: "auto" });
      });
    }
  }
}

onMounted(() => {
  void loadManual();
});

function scrollToSection(id: string) {
  const node = document.getElementById(id);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "start" });
}

if (import.meta.hot) {
  import.meta.hot.dispose((data) => {
    data.manualScrollY = window.scrollY;
  });

  import.meta.hot.accept(() => {
    // Let the remounted component restore the previous viewport position.
  });
}
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
          <p v-if="isLoading" class="text-sm text-zinc-500">Loading manual...</p>
          <p v-else-if="loadError" class="text-sm text-red-600">{{ loadError }}</p>
          <article
            v-else
            class="prose dark:prose-invert prose-emerald max-w-none prose-headings:scroll-mt-28"
            v-html="html"
          />
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
