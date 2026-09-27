<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { IconBrandDiscordFilled, IconBrandGithubFilled } from "@tabler/icons-vue";
import logoImageSc from "../assets/tools/logo_imagesc.png";
import { DISCORD_LINK, GITHUB_REPO, IMAGESC_LINK } from "../siteData";

const ORGS = [
  {
    name: "Guangdong Institute of Intelligence Science and Technology",
    logo: "/imgs/logo_gdiist.png",
    link: "https://www.gdiist.cn/",
  },
  { name: "The Hong Kong Polytechnic University", logo: "/imgs/logo_polyu.png", link: "https://www.polyu.edu.hk/" },
  { name: "Beijing Normal University", logo: "/imgs/logo_bnu.png", link: "https://www.bnu.edu.cn/" },
];

function openNewWindow(link: string) {
  window.open(link, "_blank");
}
</script>

<template>
  <section class="py-12 md:py-20">
    <div class="mx-auto w-full max-w-6xl px-6 md:px-10">
      <h2 class="text-2xl font-bold md:text-3xl">About</h2>

      <h3 class="mt-8 text-2xl font-semibold">Institutes</h3>
      <div class="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <div
          v-for="org in ORGS"
          :key="org.name"
          class="grid place-items-center rounded-2xl border border-zinc-200/60 bg-white/70 p-7 shadow-sm ring-1 ring-black/5 cursor-pointer"
          @click="openNewWindow(org.link)"
        >
          <div class="h-40 flex items-center">
            <img :src="org.logo" :alt="org.name" class="object-contain" />
          </div>

          <div class="mt-3 max-w-[320px] text-center text-base whitespace-nowrap overflow-hidden" :title="org.name">
            {{ org.name }}
          </div>
        </div>
      </div>

      <h3 class="mt-10 text-2xl font-semibold">Contact</h3>
      <div class="mt-4 grid max-w-[1020px] grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div
          class="grid place-items-center rounded-2xl border border-zinc-200/60 bg-white/70 p-7 text-center shadow-sm ring-1 ring-black/5"
        >
          <span class="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-zinc-900 text-white shadow-sm">
            <IconBrandGithubFilled class="h-10 w-10" aria-hidden="true" />
          </span>

          <a
            :href="GITHUB_REPO"
            target="_blank"
            rel="noreferrer"
            title="View source code on GitHub"
            class="mt-4 text-base font-semibold text-zinc-900 underline decoration-zinc-300 underline-offset-4 hover:decoration-zinc-900"
          >
            CopilotJ on GitHub
          </a>
          <p class="mt-3 text-sm leading-6 text-zinc-600">
            Star the repo, report issues, contribute code, and explore the source code of the CopilotJ project.
          </p>
        </div>

        <div
          class="grid place-items-center rounded-2xl border border-zinc-200/60 bg-white/70 p-7 text-center shadow-sm ring-1 ring-black/5"
        >
          <span class="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-[#5865F2] text-white shadow-sm">
            <IconBrandDiscordFilled class="h-10 w-10" aria-hidden="true" />
          </span>

          <a
            :href="DISCORD_LINK"
            target="_blank"
            rel="noreferrer"
            title="Join Discord community"
            class="mt-4 text-base font-semibold text-zinc-900 underline decoration-zinc-300 underline-offset-4 hover:decoration-[#5865F2]"
          >
            Join the CopilotJ Discord
          </a>
          <p class="mt-3 text-sm leading-6 text-zinc-600">
            Use the community server for updates, discussion, troubleshooting, and feedback around the CopilotJ project.
          </p>
        </div>

        <div
          class="grid place-items-center rounded-2xl border border-zinc-200/60 bg-white/70 p-7 text-center shadow-sm ring-1 ring-black/5"
        >
          <span
            class="inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-white shadow-sm ring-1 ring-black/5"
          >
            <img :src="logoImageSc" alt="Image.sc Forum" class="h-10 w-10 object-contain" />
          </span>

          <a
            :href="IMAGESC_LINK"
            target="_blank"
            rel="noreferrer"
            title="Ask questions on the Image.sc Forum"
            class="mt-4 text-base font-semibold text-zinc-900 underline decoration-zinc-300 underline-offset-4 hover:decoration-[#0e8ea6]"
          >
            CopilotJ on Image.sc
          </a>
          <p class="mt-3 text-sm leading-6 text-zinc-600">
            Post questions and analysis discussions to the Image.sc Forum under the
            <code class="rounded bg-zinc-100 px-1 py-0.5 text-xs">copilotj</code> tag, alongside the wider bioimage
            analysis community.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
