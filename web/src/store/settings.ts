/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { computed, ref } from "vue";
import { isExplicit, isUseServer } from "../apis";
import type { ExplicitModel, ThreadConfigModel, ThreadConfigQuery } from "../apis";
import { useConfig } from "./config";

// Empty explicit model sent when no model is configured, so the backend rejects
// it with "No model configured" instead of silently falling back to the server
// model. "Use Default Model" must never be enabled (strict BYO). TODO: temporarily disabled
const EMPTY_MODEL: ExplicitModel = { name: "", api_key: null, base_url: null, provider: null };

export const useSettings = defineStore("settings", () => {
  const expandSidebar = ref(false);
  const autoScroll = ref(true);

  const model = ref<ThreadConfigModel | null>(null);

  // The model slot sent to the backend. `model` is the union ({use_server:true}
  // or explicit); the VLM mirrors the main slot when "use main model" is on.
  const value = computed<ThreadConfigQuery>(() => {
    const cfg = useConfig();
    const vlm = cfg.data.vlm;
    let resolvedVlm: ThreadConfigModel | null = null;

    if (cfg.data.visionEnabled) {
      if (vlm.useMainModel) {
        // VLM mirrors the main slot: use_server if main is use_server, else the
        // explicit main model (or null when no main model is chosen).
        resolvedVlm = model.value;
      } else if (vlm.model) {
        resolvedVlm = { name: vlm.model, api_key: vlm.api_key, base_url: vlm.base_url, provider: vlm.provider };
      }
    }

    return {
      // model: model.value,
      // Always send an explicit model (never null / use_server) — strict BYO. TODO: temporarily disabled
      model: isExplicit(model.value) ? model.value : EMPTY_MODEL,
      vlm: resolvedVlm,
      vision_enabled: cfg.data.visionEnabled,
      proxy: cfg.data.proxy,
      tavily_api_key: cfg.data.tavilyApiKey,
      // "Auto Save" disabled; kb_autosave is omitted so the server's COPILOTJ_KB_AUTOSAVE env is the
      // sole authority (no override is sent, regardless of backend version). TODO: temporarily disabled
      // kb_autosave: cfg.data.kbAutosave,
    };
  });

  // Concrete model for DISPLAY / "is a model configured?" checks: resolves a
  // {use_server:true} choice to the server's actual model (or null if the server
  // has none).
  const effectiveModel = computed<ExplicitModel | null>(() => {
    const cfg = useConfig();
    const m = model.value;
    return m && isUseServer(m) ? cfg.serverModel : m;
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

  return {
    expandSidebar,
    autoScroll,
    value,
    model,
    effectiveModel,
    toggleAutoScroll,
    toggleExpandSidebar,
    setModel,
    reset,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useSettings, import.meta.hot));
}
