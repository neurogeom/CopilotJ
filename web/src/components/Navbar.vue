<script setup lang="ts">
import { useRouter } from "vue-router";
import { computed } from "vue";

const router = useRouter();
const currentRoute = computed(() => router.currentRoute.value.path);
const links = [
  { name: "Manual", path: "/manual" },
  { name: "About", path: "/about" },
];
</script>

<template>
  <div class="backdrop-blur-xs">
    <div class="max-w-6xl mx-auto px-4 py-3 flex justify-between items-center">
      <h1
        class="text-xl font-bold cursor-pointer hover:text-primary-500 transition-colors"
        :class="currentRoute === '/home' ? 'text-primary-500' : 'text-gray-900'"
        @click="router.push('/home')"
      >
        CopilotJ
      </h1>

      <div class="flex gap-4">
        <button
          v-for="link in links"
          :key="link.path"
          :class="[
            'px-3 py-2 rounded-md text-sm font-medium cursor-pointer transition-colors',
            currentRoute === link.path
              ? 'bg-violet-100 text-violet-500'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100',
          ]"
          @click="router.push(link.path)"
        >
          {{ link.name }}
        </button>

        <button
          class="px-3 py-2 rounded-md text-sm font-medium cursor-pointer transition-colors bg-primary-500 text-white hover:bg-primary-600"
          @click="router.push('/chat')"
        >
          Chat
        </button>
      </div>
    </div>
  </div>
</template>
