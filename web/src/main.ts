/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { definePreset } from "@primeuix/themes";
import Aura from "@primeuix/themes/aura";
import { createPinia } from "pinia";
import PrimeVue from "primevue/config";
import ConfirmationService from "primevue/confirmationservice";
import Tooltip from "primevue/tooltip";
import { createApp } from "vue";
import { createRouter, createWebHashHistory } from "vue-router";
import App from "./App.vue";
import "./style.css";

const app = createApp(App);

const pinia = createPinia();
app.use(pinia);

const pageRoutes = [
  { path: "/home", component: () => import("./views/Home.vue") },
  { path: "/manual", component: () => import("./views/Manual.vue") },
  { path: "/about", component: () => import("./views/About.vue") },
  { path: "/chat", component: () => import("./views/Chat.vue") },
];

const defaultRoute = import.meta.env.VITE_DEFAULT_ROUTE || "/chat";
const isValidPath = pageRoutes.map((r) => r.path).includes(defaultRoute);
const routes = [
  ...pageRoutes,
  {
    path: "/",
    redirect: isValidPath ? defaultRoute : "/chat",
  },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
app.use(router);

const CopilotjPreset = definePreset(Aura, {
  semantic: {
    primary: {
      50: "{violet.50}",
      100: "{violet.100}",
      200: "{violet.200}",
      300: "{violet.300}",
      400: "{violet.400}",
      500: "{violet.500}",
      600: "{violet.600}",
      700: "{violet.700}",
      800: "{violet.800}",
      900: "{violet.900}",
      950: "{violet.950}",
    },
  },
});
app.use(PrimeVue, {
  theme: {
    preset: CopilotjPreset,
    options: {
      prefix: "p",
      darkModeSelector: ".dark",
      cssLayer: false,
    },
  },
});
app.use(ConfirmationService);
app.directive("tooltip", Tooltip);

app.mount("#app");
