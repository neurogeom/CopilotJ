/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import { isExplicit } from "../apis";
import type { ExplicitModel, ThreadConfigModel } from "../apis";
import { inferProvider } from "../composables";

const STORAGE_KEY = "copilotj_config";

export interface ConfigData {
  defaultModel: ThreadConfigModel | null;
  vlm: {
    model: string | null;
    api_key: string | null;
    base_url: string | null;
    provider: string | null;
    useMainModel: boolean;
  };
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  // null = user hasn't chosen — defer to the server's COPILOTJ_VISION_ENABLED.
  // true/false = explicit user choice (Settings Save / Wizard), authoritative.
  visionEnabled: boolean | null;
  userAgreement: boolean;
}

function defaultConfig(): ConfigData {
  return {
    defaultModel: null,
    vlm: { model: null, api_key: null, base_url: null, provider: null, useMainModel: true },
    proxy: null,
    tavilyApiKey: null,
    kbAutosave: false,
    visionEnabled: null,
    userAgreement: false,
  };
}

function loadFromStorage(): ConfigData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      const source = typeof parsed === "object" && parsed !== null ? parsed : {};
      // Drop the short-lived legacy `visionPrefSet` key (it never shipped to
      // main). A chosen value (true) is kept; an unchosen one (false) becomes
      // null (defer to server). Configs without it keep visionEnabled as-is.
      const { visionPrefSet: legacyPrefSet, ...rest } = source as Record<string, unknown>;
      const config = { ...defaultConfig(), ...(rest as Partial<ConfigData>) };
      if (legacyPrefSet !== undefined) {
        config.visionEnabled = legacyPrefSet ? config.visionEnabled : null;
      }
      return migrateConfig(config);
    }
  } catch {
    // ignore parse errors
  }
  return defaultConfig();
}

/** Migrate legacy provider values and infer missing providers from model names. */
function migrateConfig(config: ConfigData): ConfigData {
  const dm = config.defaultModel;
  if (dm && isExplicit(dm)) {
    if (dm.api_key === null && !dm.name.startsWith("ollama/")) {
      // Legacy "null api_key → borrow the server's key" hack: previously converted
      // to the explicit "use the server's model" choice.
      // "Use Default Model" disabled (strict BYO): clear the choice instead. TODO: temporarily disabled
      // config.defaultModel = { use_server: true };
      config.defaultModel = null;
    } else {
      if (dm.provider === "google") dm.provider = "gemini";
      if (!dm.provider && dm.name) dm.provider = inferProvider(dm.name);
    }
  }
  if (config.vlm) {
    if (config.vlm.provider === "google") {
      config.vlm.provider = "gemini";
    }
    if (!config.vlm.provider && config.vlm.model) {
      config.vlm.provider = inferProvider(config.vlm.model);
    }
  }
  return config;
}

function saveToStorage(config: ConfigData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export const useConfig = defineStore("config", () => {
  const data = ref<ConfigData>(loadFromStorage());

  // Server model from /api/config — NOT persisted to localStorage. Display-only
  // (resolved name/endpoint); never sent back as a payload.
  const serverModel = ref<ExplicitModel | null>(null);

  function setServerModel(model: ExplicitModel | null) {
    serverModel.value = model;
  }

  // Server's vision_enabled from /api/config — NOT persisted. Display-only: lets
  // the UI render the effective Vision state when visionEnabled is null (user
  // hasn't chosen). Never sent back as a payload.
  const serverVisionEnabled = ref<boolean | null>(null);

  function setServerVisionEnabled(enabled: boolean | null) {
    serverVisionEnabled.value = enabled;
  }

  function persist() {
    saveToStorage(data.value);
  }

  function setDefaultModel(model: ThreadConfigModel | null) {
    data.value.defaultModel = model;
    persist();
  }

  function setVlm(vlm: ConfigData["vlm"]) {
    data.value.vlm = vlm;
    persist();
  }

  function setProxy(proxy: string | null) {
    data.value.proxy = proxy;
    persist();
  }

  function setTavilyApiKey(key: string | null) {
    data.value.tavilyApiKey = key;
    persist();
  }

  function setKbAutosave(enabled: boolean) {
    data.value.kbAutosave = enabled;
    persist();
  }

  function setVisionEnabled(enabled: boolean) {
    // Explicit user choice (Settings Save / Wizard). Afterwards the value is
    // authoritative; reloads never flip it. null (defer) is restored by reset().
    data.value.visionEnabled = enabled;
    persist();
  }

  function setUserAgreement(accepted: boolean) {
    data.value.userAgreement = accepted;
    persist();
  }

  function reset() {
    data.value = defaultConfig();
    localStorage.removeItem(STORAGE_KEY);
  }

  return {
    data,
    serverModel,
    serverVisionEnabled,
    setServerModel,
    setServerVisionEnabled,
    setDefaultModel,
    setVlm,
    setProxy,
    setTavilyApiKey,
    setKbAutosave,
    setVisionEnabled,
    setUserAgreement,
    reset,
    persist,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useConfig, import.meta.hot));
}
