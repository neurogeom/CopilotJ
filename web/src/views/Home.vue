<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { ref } from "vue";
import { RouterLink } from "vue-router";
import {
  IconArrowRight,
  IconCheck,
  IconExternalLink,
  IconBrandDiscordFilled,
  IconBrandGithubFilled,
} from "@tabler/icons-vue";
import Demos from "../components/Demos.vue";
import Logo from "../components/Logo.vue";
import { BACKGROUND, CONTRIBUTIONS, DISCORD_LINK, FEATURES, GITHUB_REPO, ORGS, TOOLS, SLOGAN } from "../siteData";

const demosRef = ref<HTMLElement | null>(null);

function scrollToDemos() {
  if (!demosRef.value) return;
  const top = demosRef.value.getBoundingClientRect().top + window.scrollY - 80;
  window.scrollTo({ top, behavior: "smooth" });
}

function openNewWindow(link: string) {
  window.open(link, "_blank");
}
</script>

<template>
  <div class="w-full flex flex-col gap-18 md:gap-24">
    <section class="py-12 md:px-10 lg:py-24 text-center">
      <h1 class="max-w-4xl mx-auto text-3xl font-bold leading-tight md:text-5xl">
        <span class="bg-linear-to-r from-blue-500 to-primary-500 bg-clip-text text-transparent">CopilotJ</span>: A
        Conversational Multi-agent System for Intelligent and Efficient Bioimage Analysis
      </h1>

      <p class="mx-auto mt-8 max-w-4xl text-base leading-8 text-zinc-600 md:text-xl">
        {{ SLOGAN }}
      </p>

      <div class="mt-12 flex flex-wrap items-center justify-center gap-3 font-medium">
        <button
          @click="scrollToDemos"
          class="inline-flex items-center gap-2 rounded-full bg-primary-500 px-8 py-2.5 text-sm text-white cursor-pointer shadow transition-colors hover:bg-primary-800"
        >
          Demos
        </button>

        <RouterLink
          to="/manual"
          class="inline-flex items-center gap-2 rounded-full border border-zinc-300/60 px-6 py-2.5 transition-colors text-sm text-zinc-800 hover:bg-zinc-50"
          >Get Started <IconArrowRight class="h-5"
        /></RouterLink>

        <a
          :href="GITHUB_REPO"
          target="_blank"
          rel="noreferrer"
          title="View source code on GitHub"
          class="inline-flex items-center gap-2 rounded-full border border-zinc-300/60 px-6 py-2.5 text-sm text-zinc-800 hover:bg-zinc-50"
          >Source Code <IconExternalLink class="h-5"
        /></a>
      </div>
    </section>

    <section class="max-w-5xl mx-auto flex">
      <div
        v-for="item in FEATURES"
        :key="`features-${item.label}`"
        class="flex-1 my-2 px-6 not-last:border-r border-gray-200"
      >
        <div class="my-4 text-3xl font-bold text-zinc-900 flex items-center">
          <span class="mr-4 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/90">
            <IconCheck class="h-4 w-4 text-white" />
          </span>
          {{ item.label }}
        </div>
        <div class="mt-1 text-sm leading-6 text-zinc-600">{{ item.description }}</div>
      </div>
    </section>

    <section class="max-w-6xl mx-auto px-4 flex gap-3">
      <div class="flex-1 rounded-2xl border p-6 border-zinc-200/60 bg-white/70 shadow-sm ring-1 ring-black/5">
        <h3 class="text-xl font-semibold md:text-2xl">Background</h3>
        <div class="mt-3 space-y-3">
          <p v-for="(t, i) in BACKGROUND" :key="`bg-${i}`" class="text-base leading-7 text-zinc-700">
            {{ t }}
          </p>
        </div>
      </div>

      <div class="flex-1 rounded-2xl border p-6 border-zinc-200/60 bg-white/70 shadow-sm ring-1 ring-black/5">
        <h3 class="text-xl font-semibold md:text-2xl">What makes CopilotJ different</h3>

        <div class="mt-3 list-disc space-y-3 leading-7 text-zinc-700">
          <div v-for="(t, i) in CONTRIBUTIONS" :key="`contrib-${i}`">
            <h4 class="font-medium">{{ t.title }}</h4>

            <div class="pl-5">{{ t.description }}</div>
          </div>
        </div>
      </div>
    </section>

    <div ref="demosRef" class="w-full max-w-6xl mx-auto px-4">
      <Demos />
    </div>

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">Software architecture of CopilotJ</h2>

      <div
        class="mt-4 rounded-2xl border border-zinc-200/60 bg-white/70 p-4 shadow-sm ring-1 ring-black/5 grid grid-cols-1 gap-5 lg:grid-cols-3"
      >
        <div class="lg:col-span-2">
          <img
            src="/imgs/supplementary_figure6.png"
            alt="Software architecture of CopilotJ"
            class="w-full rounded-xl border border-zinc-200/60"
          />
        </div>

        <div>
          CopilotJ comprises three major components, a web-based frontend, a multi-agent backend, and an ImageJ/Fiji
          plugin called the CopilotJ Bridge. The user interacts with CopilotJ through the chat interface, which sends
          requests to the multi-agent runtime and receives streaming responses via RESTful APIs. The backend
          orchestrates agents and tools, integrates Python and deep-learning environments with a local database,
          connects to multiple LLM providers, and retrieves community knowledge sources. The agents communicate with
          ImageJ/Fiji through the CopilotJ Bridge via a bidirectional WebSocket API, enabling exchange of system status
          and commands
        </div>
      </div>
    </section>

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">CopilotJ workflow</h2>

      <div
        class="mt-4 rounded-2xl border border-zinc-200/60 bg-white/70 p-4 shadow-sm ring-1 ring-black/5 grid grid-cols-1 gap-5 lg:grid-cols-3"
      >
        <div class="lg:col-span-2">
          <img
            src="/imgs/figure1.png"
            alt="CopilotJ architecture"
            class="w-full rounded-xl border border-zinc-200/60"
          />
        </div>

        <ul class="mt-2 list-disc space-y-2 pl-5">
          <li>
            Summarizes the conversational workflow from user intent to execution, validation, and output generation
          </li>
          <li>Shows how the Leader, Kernel, Research, and Enhancement agents collaborate during runtime</li>
          <li>
            Connects planning, tool execution, knowledge retrieval, and reporting in a single end-to-end system view
          </li>
        </ul>
      </div>
    </section>

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">Applications</h2>
      <div
        class="mt-4 rounded-2xl border border-zinc-200/60 bg-white/70 p-4 shadow-sm ring-1 ring-black/5 grid grid-cols-1 gap-5 lg:grid-cols-3"
      >
        <div class="lg:col-span-2">
          <img
            src="/imgs/figure2a.png"
            alt="Applications and benchmarking"
            class="w-full rounded-xl border border-zinc-200/60"
          />
        </div>

        <ul class="mt-2 list-disc space-y-2 pl-5">
          <li>Showcase of twelve exemplar analysis tasks performed using CopilotJ</li>
          <li>For each subpanel, the task ID and task name are displayed at the top</li>
        </ul>
      </div>
    </section>

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">
        Extended ImageJ envorionment for deep learning and scientific computing
      </h2>
      <div
        class="mt-4 rounded-2xl border border-zinc-200/60 bg-white/70 p-4 shadow-sm ring-1 ring-black/5 grid grid-cols-1 gap-5 lg:grid-cols-3"
      >
        <div class="lg:col-span-2">
          <img
            src="/imgs/figure2ef.png"
            alt="Applications and benchmarking"
            class="w-full rounded-xl border border-zinc-200/60"
          />
        </div>

        <ul class="mt-2 list-disc space-y-2 pl-5">
          <li>Left, access to deep-learning tools through Extended ImageJ Environment</li>
          <li>
            Center, illustrative interface showing a user interacting with CopilotJ to invoke deep-learning models for
            specific analysis goals
          </li>
          <li>
            Right, Input images and corresponding results for image super-resolution, cell classification and cell
            profiling (top to bottom)
          </li>
        </ul>
      </div>
    </section>

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">Benchmarking</h2>
      <div
        class="mt-4 rounded-2xl border border-zinc-200/60 bg-white/70 p-4 shadow-sm ring-1 ring-black/5 grid grid-cols-1 gap-5 lg:grid-cols-3"
      >
        <div class="lg:col-span-2">
          <img
            src="/imgs/figure2d.png"
            alt="Applications and benchmarking"
            class="w-full rounded-xl border border-zinc-200/60"
          />
        </div>

        <ul class="mt-2 list-disc space-y-2 pl-5">
          <li>
            Reproducibility rate of CopilotJ across five representative tasks using different LLMs (Google Gemini 2.5
            Pro, OpenAI GPT-4.1 and z.ai GLM 4.6)
          </li>
          <li>
            Completion time and the number of mouse/keyboard interactions during task completion across five
            representative tasks. Bars in orange, grey, and intermediate shades denote LLM-guided ImageJ, whereas purple
            bars denote CopilotJ; Error bars indicate 95% confidence intervals; Statistical significance was assessed
            using a linear mixed-effects mode. P < 0.05 (*), P < 0.01 (**), P < 0.001 (***). Results are shown
            separately for beginner and experienced users
          </li>
          <li>
            NASA-TLX questionnaire score. Orange bars denote LLM-guided ImageJ and purple bars denote CopilotJ;
            Statistical significance was assessed using a t-test. P < 0.05 (*), P < 0.01 (**), P < 0.001 (***). For both
            panels, results are shown separately for beginner and experienced users.
          </li>
        </ul>
      </div>
    </section>

    <!-- <Community /> -->

    <section class="max-w-6xl mx-auto px-4">
      <h2 class="text-2xl font-bold md:text-3xl">Integrated tools</h2>

      <div class="mt-4 grid grid-cols-4 md:grid-cols-6 gap-3">
        <div
          v-for="tool in TOOLS"
          :key="tool.name"
          class="tool-card group p-4 bg-gray-200 rounded-lg flex items-center justify-center cursor-pointer transition-all duration-300 hover:bg-white hover:shadow-lg hover:-translate-y-1 hover:scale-105"
        >
          <img
            :src="tool.logo"
            :alt="tool.name"
            class="max-h-12 w-auto object-contain transition-transform duration-300 group-hover:scale-110"
          />
        </div>

        <div
          class="tool-card group p-4 bg-linear-to-br from-gray-200 to-gray-300 rounded-lg flex items-center justify-center cursor-pointer transition-all duration-300 hover:from-primary-500 hover:to-blue-600 hover:shadow-lg hover:-translate-y-1 hover:scale-105"
        >
          <span class="text-2xl text-gray-500 transition-colors duration-300 group-hover:text-white">and more...</span>
        </div>
      </div>
    </section>

    <section class="pt-36 bg-gray-200">
      <div class="flex flex-col items-center">
        <Logo class="h-32 w-32" />

        <h3 class="my-8 text-5xl text-center font-medium leading-18">Meet CopilotJ</h3>

        <RouterLink
          to="/manual"
          class="px-6 py-2.5 inline-flex items-center gap-2 rounded-full border border-primary-500 transition-colors text-sm text-white bg-primary-500 hover:bg-primary-600"
          >Get Started <IconArrowRight class="h-5"
        /></RouterLink>
      </div>

      <div class="max-w-6xl mx-auto px-4 mt-36 flex justify-between items-center">
        <div class="flex-1 flex gap-8">
          <div
            v-for="org in ORGS"
            :key="org.name"
            class="h-max-30 max-w-50 flex items-center cursor-pointer"
            :title="`Visit ${org.name} website`"
            @click="openNewWindow(org.link)"
          >
            <img :src="org.logo" :alt="org.name" class="object-contain" />
          </div>
        </div>

        <div class="flex gap-4">
          <IconBrandDiscordFilled
            class="h-8 w-8 text-zinc-500 hover:text-[#5865F2] transition-colors cursor-pointer"
            title="Join Discord community"
            @click="openNewWindow(DISCORD_LINK)"
          />

          <IconBrandGithubFilled
            class="h-8 w-8 text-zinc-500 hover:text-black transition-colors cursor-pointer"
            title="View GitHub repository"
            @click="openNewWindow(GITHUB_REPO)"
          />
        </div>
      </div>

      <div class="my-4 text-sm text-zinc-500 text-center">Copyright ©2026. All Rights Reserved</div>
    </section>
  </div>
</template>

<style scoped>
.tool-card {
  animation: fadeInUp 0.6s ease-out backwards;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Stagger animation for tool cards */
.tool-card:nth-child(1) {
  animation-delay: 0.05s;
}
.tool-card:nth-child(2) {
  animation-delay: 0.1s;
}
.tool-card:nth-child(3) {
  animation-delay: 0.15s;
}
.tool-card:nth-child(4) {
  animation-delay: 0.2s;
}
.tool-card:nth-child(5) {
  animation-delay: 0.25s;
}
.tool-card:nth-child(6) {
  animation-delay: 0.3s;
}
.tool-card:nth-child(7) {
  animation-delay: 0.35s;
}
.tool-card:nth-child(8) {
  animation-delay: 0.4s;
}
.tool-card:nth-child(9) {
  animation-delay: 0.45s;
}
.tool-card:nth-child(10) {
  animation-delay: 0.5s;
}
.tool-card:nth-child(11) {
  animation-delay: 0.55s;
}
.tool-card:nth-child(12) {
  animation-delay: 0.6s;
}
.tool-card:nth-child(13) {
  animation-delay: 0.65s;
}
.tool-card:nth-child(14) {
  animation-delay: 0.7s;
}
.tool-card:nth-child(15) {
  animation-delay: 0.75s;
}
.tool-card:nth-child(16) {
  animation-delay: 0.8s;
}
.tool-card:nth-child(17) {
  animation-delay: 0.85s;
}
.tool-card:nth-child(18) {
  animation-delay: 0.9s;
}
.tool-card:nth-child(19) {
  animation-delay: 0.95s;
}
.tool-card:nth-child(20) {
  animation-delay: 1s;
}
.tool-card:nth-child(21) {
  animation-delay: 1.05s;
}
.tool-card:nth-child(22) {
  animation-delay: 1.1s;
}
.tool-card:nth-child(23) {
  animation-delay: 1.15s;
}
.tool-card:nth-child(24) {
  animation-delay: 1.2s;
}
.tool-card:nth-child(25) {
  animation-delay: 1.25s;
}
.tool-card:nth-child(26) {
  animation-delay: 1.3s;
}
.tool-card:nth-child(27) {
  animation-delay: 1.35s;
}
.tool-card:nth-child(28) {
  animation-delay: 1.4s;
}
.tool-card:nth-child(29) {
  animation-delay: 1.45s;
}
.tool-card:nth-child(30) {
  animation-delay: 1.5s;
}
</style>
