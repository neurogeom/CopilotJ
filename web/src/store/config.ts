/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { acceptHMRUpdate, defineStore } from "pinia";
import { ref } from "vue";
import type { ThreadConfigModel } from "../apis";

const STORAGE_KEY = "copilotj_config";

export interface ConfigData {
  defaultModel: ThreadConfigModel | null;
  vlm: {
    model: string | null;
    api_key: string | null;
    base_url: string | null;
    useMainModel: boolean;
  };
  proxy: string | null;
  tavilyApiKey: string | null;
  kbAutosave: boolean;
  visionEnabled: boolean;
}

function defaultConfig(): ConfigData {
  return {
    defaultModel: null,
    vlm: { model: null, api_key: null, base_url: null, useMainModel: true },
    proxy: null,
    tavilyApiKey: null,
    kbAutosave: false,
    visionEnabled: false,
  };
}

function loadFromStorage(): ConfigData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      return { ...defaultConfig(), ...JSON.parse(raw) };
    }
  } catch {
    // ignore parse errors
  }
  return defaultConfig();
}

function saveToStorage(config: ConfigData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
}

export const useConfig = defineStore("config", () => {
  const data = ref<ConfigData>(loadFromStorage());

  // Server model from /api/config — NOT persisted to localStorage.
  // Refreshed on every app load so it always reflects the backend's actual model.
  const serverModel = ref<ThreadConfigModel | null>(null);

  function setServerModel(model: ThreadConfigModel | null) {
    serverModel.value = model;
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
    data.value.visionEnabled = enabled;
    persist();
  }

  function reset() {
    data.value = defaultConfig();
    localStorage.removeItem(STORAGE_KEY);
  }

  return {
    data,
    serverModel,
    setServerModel,
    setDefaultModel,
    setVlm,
    setProxy,
    setTavilyApiKey,
    setKbAutosave,
    setVisionEnabled,
    reset,
    persist,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useConfig, import.meta.hot));
}
