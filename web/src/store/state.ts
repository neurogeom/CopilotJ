/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import { getBaseUrl, testApiConnection } from "../apis/base";

/** The tab values used by the Settings dialog. Shared so any caller can open
 * Settings on the tab that matches the problem the user needs to fix. */
export type SettingsTab = "notice" | "base" | "model" | "vision" | "pref";

export const useSystemState = defineStore("state", () => {
  const showSettings = ref(false);
  const showManageAgents = ref(false);
  const wizardMode = ref(false);

  /** Active Settings tab — single source of truth for cross-tab jumps and
   * chat-error navigation. Bound two-way by Settings.vue. */
  const settingsTab = ref<SettingsTab>("notice");

  const backendReachable = ref<boolean | null>(null);
  const connectionWarningDismissed = ref(false);

  async function testBackendConnection() {
    const rawUrl = getBaseUrl().replace(/\/api$/, "");
    backendReachable.value = await testApiConnection(rawUrl);
  }

  /** Open the Settings dialog, optionally landing on a specific tab. */
  function openSettings(tab?: SettingsTab) {
    if (tab) settingsTab.value = tab;
    showSettings.value = true;
  }

  return {
    showManageAgents,
    showSettings,
    settingsTab,
    wizardMode,
    backendReachable,
    connectionWarningDismissed,
    openSettings,
    testBackendConnection,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSystemState, import.meta.hot));
}
