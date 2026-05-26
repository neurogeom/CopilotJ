/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const STORAGE_KEY = "copilotj_api_base_url";

export function getBaseUrl(): string {
  const customUrl = localStorage.getItem(STORAGE_KEY);
  const apiBaseUrl = customUrl || import.meta.env.VITE_API_BASE_URL || "";
  const normalized = apiBaseUrl.replace(/\/+$/, "");
  return normalized ? `${normalized}/api` : "/api";
}

export function setApiBaseUrl(url: string): void {
  if (url) {
    localStorage.setItem(STORAGE_KEY, url);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export const isApiBaseConfigurable =
  typeof import.meta.env.VITE_CONFIGURABLE_API_BASE !== "undefined"
    ? !!import.meta.env.VITE_CONFIGURABLE_API_BASE
    : true; // enabled by default if not set

export function isApiBaseConfigured(): boolean {
  const customUrl = localStorage.getItem(STORAGE_KEY);
  const apiBaseUrl = customUrl || import.meta.env.VITE_API_BASE_URL || "";
  return apiBaseUrl.length > 0;
}

export async function testApiConnection(url: string): Promise<boolean> {
  try {
    const normalized = url.replace(/\/+$/, "");
    const pingUrl = normalized ? `${normalized}/api/ping` : "/api/ping";
    const resp = await fetch(pingUrl, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      return false;
    }

    const data = await resp.text();
    return data === "pong";
  } catch {
    return false;
  }
}
