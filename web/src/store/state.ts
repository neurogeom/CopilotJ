/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import { getBaseUrl, testApiConnection } from "../apis/base";
import { getServerVersion, isApiVersionCompatible } from "../apis/version";

export const useSystemState = defineStore("state", () => {
  const showSettings = ref(false);
  const showManageAgents = ref(false);
  const wizardMode = ref(false);

  const backendReachable = ref<boolean | null>(null);
  const connectionWarningDismissed = ref(false);

  // Frontend <-> server protocol version check (issue #68)
  const serverApiVersion = ref<string | null>(null);
  const apiVersionStatus = ref<"unknown" | "compatible" | "incompatible" | null>(null);
  const apiVersionWarningDismissed = ref(false);

  async function testBackendConnection() {
    const rawUrl = getBaseUrl().replace(/\/api$/, "");
    backendReachable.value = await testApiConnection(rawUrl);
  }

  async function checkApiVersion() {
    try {
      const { api_version } = await getServerVersion();
      serverApiVersion.value = api_version;
      apiVersionStatus.value = isApiVersionCompatible(api_version) ? "compatible" : "incompatible";
    } catch {
      // Endpoint missing (old server), network error, or unparseable → unknown.
      serverApiVersion.value = null;
      apiVersionStatus.value = "unknown";
    }
  }

  return {
    showManageAgents,
    showSettings,
    wizardMode,
    backendReachable,
    connectionWarningDismissed,
    testBackendConnection,
    serverApiVersion,
    apiVersionStatus,
    apiVersionWarningDismissed,
    checkApiVersion,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSystemState, import.meta.hot));
}
