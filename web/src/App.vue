<!--
SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.

SPDX-License-Identifier: Apache-2.0
-->

<script setup lang="ts">
import { useRouter, RouterView } from "vue-router";
import { computed } from "vue";
import Navbar from "./components/Navbar.vue";
import HeroPattern from "./components/HeroPattern.vue";

const isRoot = computed(() => currentRoute.value === "/");

const router = useRouter();
const currentRoute = computed(() => router.currentRoute.value.path);
</script>

<template>
  <div class="relative min-h-screen flex flex-col">
    <!-- Navigation -->
    <nav v-if="!isRoot && currentRoute !== '/chat'" class="w-full sticky top-0 border-b border-gray-200 z-50">
      <Navbar />
    </nav>

    <!-- Main content -->
    <main class="flex-1">
      <RouterView />
    </main>

    <HeroPattern v-if="!isRoot" class="absolute top-0 w-full h-full -z-10 opacity-20" />
  </div>
</template>

<style lang="postcss">
@reference "./style.css";
</style>
