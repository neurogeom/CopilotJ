/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { getBaseUrl } from "./base";

/** Default Ollama host used when the caller does not supply one. */
const DEFAULT_OLLAMA_URL = "http://localhost:11434";

export interface ModelItem {
  id: string;
  label: string;
  supports_vision: boolean;
  context_window: number | null;
}

export interface ProviderModels {
  provider: string;
  /** "catalog" (LiteLLM DB) | "live" (Ollama reachable) | "unreachable" (Ollama not reachable). */
  source: string;
  models: ModelItem[];
}

/**
 * Build the `base_url` query segment for Ollama. Only forwarded when the caller
 * supplied a non-default host, so the common localhost case keeps requests clean.
 */
function ollamaParam(baseUrl?: string): string {
  const trimmed = (baseUrl ?? "").trim();
  if (trimmed && trimmed !== DEFAULT_OLLAMA_URL) {
    return `&base_url=${encodeURIComponent(trimmed)}`;
  }
  return "";
}

async function fetchModels(query: string): Promise<any> {
  const url = `${getBaseUrl()}/models${query}`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

/** List models for a single provider. `baseUrl` is only meaningful for Ollama. */
export async function listProviderModels(provider: string, baseUrl?: string): Promise<ProviderModels> {
  return fetchModels(`?provider=${encodeURIComponent(provider)}${ollamaParam(baseUrl)}`);
}

/**
 * List models for all providers at once. `ollamaBaseUrl` applies to the Ollama
 * entry only; the cloud providers come from the cached catalog regardless.
 */
export async function listModels(ollamaBaseUrl?: string): Promise<Record<string, ProviderModels>> {
  return fetchModels(ollamaParam(ollamaBaseUrl));
}
