/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { computed, ref } from "vue";
import type { ThreadConfigModel, ThreadConfigQuery } from "../apis";
import { useConfig } from "./config";

export const useSettings = defineStore("settings", () => {
  const expandSidebar = ref(false);
  const autoScroll = ref(true);

  const model = ref<ThreadConfigModel | null>(null);
  const value = computed<ThreadConfigQuery>(() => {
    const cfg = useConfig();
    const vlm = cfg.data.vlm;
    let resolvedVlm: ThreadConfigModel | null = null;

    if (cfg.data.visionEnabled) {
      if (vlm.useMainModel) {
        // Resolve main model details as the VLM so the backend
        // uses the correct model without needing a "useMainModel" concept
        resolvedVlm = model.value
          ? { name: model.value.name, api_key: model.value.api_key, base_url: model.value.base_url }
          : null;
      } else {
        resolvedVlm = { name: vlm.model!, api_key: vlm.api_key, base_url: vlm.base_url };
      }
    }

    return {
      model: model.value,
      vlm: resolvedVlm,
      vision_enabled: cfg.data.visionEnabled,
      proxy: cfg.data.proxy,
      tavily_api_key: cfg.data.tavilyApiKey,
      kb_autosave: cfg.data.kbAutosave,
    };
  });

  function toggleExpandSidebar(enable?: boolean) {
    expandSidebar.value = enable ?? !expandSidebar.value;
  }

  function toggleAutoScroll(enable?: boolean) {
    autoScroll.value = enable ?? !autoScroll.value;
  }

  function setModel(newModel: ThreadConfigModel | null) {
    model.value = newModel;
  }

  function reset() {
    expandSidebar.value = true;
    autoScroll.value = false;
  }

  return { expandSidebar, autoScroll, value, model, toggleAutoScroll, toggleExpandSidebar, setModel, reset };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSettings, import.meta.hot));
}
