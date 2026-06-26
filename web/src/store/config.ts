/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import { getServerConfig, isExplicit, isUseServer } from "../apis";
import type { ExplicitModel, ThreadConfigModel } from "../apis";
import { inferProvider } from "../composables";
import { useSettings } from "./settings";

const STORAGE_KEY = "copilotj_config";

export interface ConfigData {
  defaultModel: ThreadConfigModel | null;
  vlm: {
    model: string | null;
    api_key: string | null;
    base_url: string | null;
    provider: string | null;
    useMainModel: boolean;
    useServerVlm: boolean;
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
    vlm: { model: null, api_key: null, base_url: null, provider: null, useMainModel: true, useServerVlm: false },
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
      // Legacy "null api_key → borrow the server's key" hack: convert to the
      // explicit "use the server's model" choice.
      config.defaultModel = { use_server: true };
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

  // Server VLM from /api/config — NOT persisted. Display/availability only: the
  // Vision tab's "Use server vision model" toggle is enabled iff this is non-null.
  const serverVlm = ref<ExplicitModel | null>(null);

  function setServerVlm(vlm: ExplicitModel | null) {
    serverVlm.value = vlm;
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

  // Re-fetch /api/config from the current backend and reconcile derived state.
  // Shared by Chat.vue onMounted and the Settings soft-reconnect. On failure
  // returns false AND clears the server-derived display refs (model/VLM/vision)
  // so callers never show the old backend's capabilities after a failed
  // reconnect — mirrors the old page-reload behavior that reset everything.
  async function applyServerConfig(): Promise<boolean> {
    const settings = useSettings();
    try {
      const serverConfig = await getServerConfig();

      setServerModel(serverConfig.model);
      setServerVlm(serverConfig.vlm);

      // Model — if nothing is configured yet but the server has a model, default
      // to "use the server's model" (an explicit, persisted choice).
      if (settings.model === null && serverConfig.model !== null) {
        const useServer: ThreadConfigModel = { use_server: true };
        settings.setModel(useServer);
        setDefaultModel(useServer);
      }

      // Reconcile stale "use server" choices: if the loaded server has no model
      // for a slot, a persisted {use_server:true} / useServerVlm choice is no
      // longer valid — clear it so the UI asks for an explicit model.
      if (isUseServer(data.value.defaultModel) && !serverConfig.model) {
        setDefaultModel(null);
        settings.setModel(null);
      }
      if (data.value.vlm.useServerVlm && !serverConfig.vlm) {
        setVlm({ ...data.value.vlm, useServerVlm: false });
      }

      // Proxy
      if (data.value.proxy === null && serverConfig.proxy !== null) {
        setProxy(serverConfig.proxy);
      }

      // Vision: store the server's default for display only.
      setServerVisionEnabled(serverConfig.vision_enabled);
      return true;
    } catch {
      // Server not reachable or /api/config errored. Clear the server-derived
      // display refs so a failed reconnect never leaves the old backend's
      // model/VLM shown (the old page-reload wiped this state). Callers
      // surface the failure; we return false.
      setServerModel(null);
      setServerVlm(null);
      setServerVisionEnabled(null);
      return false;
    }
  }

  function reset() {
    data.value = defaultConfig();
    localStorage.removeItem(STORAGE_KEY);
  }

  return {
    data,
    serverModel,
    serverVlm,
    serverVisionEnabled,
    setServerModel,
    setServerVlm,
    setServerVisionEnabled,
    setDefaultModel,
    setVlm,
    setProxy,
    setTavilyApiKey,
    setKbAutosave,
    setVisionEnabled,
    setUserAgreement,
    applyServerConfig,
    reset,
    persist,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useConfig, import.meta.hot));
}
