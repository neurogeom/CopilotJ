/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import { getBaseUrl, testApiConnection } from "../apis/base";

export const useSystemState = defineStore("state", () => {
  const showSettings = ref(false);
  const showManageAgents = ref(false);

  const backendReachable = ref<boolean | null>(null);
  const connectionWarningDismissed = ref(false);

  async function testBackendConnection() {
    const rawUrl = getBaseUrl().replace(/\/api$/, "");
    backendReachable.value = await testApiConnection(rawUrl);
  }

  return { showManageAgents, showSettings, backendReachable, connectionWarningDismissed, testBackendConnection };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSystemState, import.meta.hot));
}
