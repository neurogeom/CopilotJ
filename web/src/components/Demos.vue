<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { computed, ref } from "vue";
import { SUPPLEMENTARY_VIDEOS } from "../siteData";

const active = ref(11);
const activeGroup = ref("Multi-stage pipelines");
const activeVideo = computed(
  () => SUPPLEMENTARY_VIDEOS.find((video) => video.id === active.value) ?? SUPPLEMENTARY_VIDEOS[0],
);
const groups = computed(() => {
  const map = new Map<string, any[]>();
  SUPPLEMENTARY_VIDEOS.forEach((video) => {
    const items = map.get(video.group) ?? [];
    items.push(video);
    map.set(video.group, items);
  });
  return Array.from(map.entries()).map(([name, items]) => ({ name, items, count: items.length }));
});

const currentGroup = computed(() => groups.value.find((group) => group.name === activeGroup.value) ?? groups.value[0]);

function setGroup(name: string) {
  activeGroup.value = name;
  if (!currentGroup.value.items.some((video) => video.id === active.value)) {
    active.value = currentGroup.value.items[0]?.id ?? active.value;
  }
}

function youtubeEmbed(url: string) {
  const id = url.match(/(?:v=|youtu\.be\/)([^&?/]+)/)?.[1] ?? "";
  return `https://www.youtube.com/embed/${id}`;
}
</script>

<template>
  <section class="w-full min-w-0 flex flex-col">
    <h2 class="text-2xl font-bold md:text-3xl">Demos</h2>

    <div class="w-full mt-6 flex gap-2 overflow-x-auto">
      <button
        v-for="group in groups"
        :key="group.name"
        type="button"
        class="rounded-full border px-2.5 py-1.5 text-sm transition-colors cursor-pointer whitespace-nowrap"
        :class="
          group.name === activeGroup
            ? 'border-primary-800 bg-primary-800 text-white shadow'
            : 'border-primary-300/50 bg-white/70 text-zinc-700 hover:bg-white'
        "
        @click="setGroup(group.name)"
      >
        {{ group.name }}
      </button>
    </div>

    <div class="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
      <div
        class="min-w-0 overflow-hidden rounded-3xl border border-zinc-200/60 bg-white/70 shadow-lg ring-1 ring-black/5 lg:col-span-2"
      >
        <div
          class="flex flex-col gap-2 border-b border-zinc-200/60 bg-white/80 px-5 py-3 sm:flex-row sm:items-center sm:justify-between"
        >
          <div class="min-w-0 mt-1 text-lg font-semibold break-words">{{ activeVideo.title }}</div>

          <a
            :href="activeVideo.youtubeUrl"
            target="_blank"
            rel="noreferrer"
            class="inline-flex shrink-0 items-center gap-2 px-1 py-1.5 font-medium transition-colors text-zinc-500 hover:text-zinc-600 hover:underline"
          >
            Open on YouTube
          </a>
        </div>

        <div class="bg-black">
          <iframe
            :key="activeVideo.youtubeUrl"
            class="aspect-video w-full"
            :src="youtubeEmbed(activeVideo.youtubeUrl)"
            :title="activeVideo.title"
            loading="lazy"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowfullscreen
            referrerpolicy="strict-origin-when-cross-origin"
          />
        </div>
      </div>

      <div class="min-w-0 mt-4">
        <div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
          <h3 class="min-w-0 text-xl font-semibold break-words">{{ activeGroup }}</h3>
          <div class="shrink-0 text-sm text-zinc-500">{{ currentGroup.items.length }} videos</div>
        </div>

        <div class="mt-4 flex flex-col gap-3">
          <button
            v-for="video in currentGroup.items"
            :key="video.id"
            type="button"
            class="rounded-2xl border px-4 py-3 text-left shadow-sm ring-1 ring-black/5 transition"
            :class="
              video.id === activeVideo.id
                ? 'border-primary-800 bg-primary-800 text-white'
                : 'border-primary-200/60 bg-white/70 text-primary-900'
            "
            @click="active = video.id"
          >
            <div class="my-1 break-words text-base font-semibold leading-6 transition-colors">{{ video.title }}</div>

            <p
              class="line-clamp-3 break-words text-sm leading-6 transition-all"
              :class="video.id === activeVideo.id ? 'max-h-64 text-zinc-200' : 'max-h-0 text-zinc-600'"
            >
              {{ video.summary }}
            </p>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
