/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

const DEFAULT_API_ORIGIN = "http://127.0.0.1:8786";
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_ORIGIN;
const normalizedApiBaseUrl = apiBaseUrl.replace(/\/+$/, "");
const BASE_URL = normalizedApiBaseUrl.endsWith("/api")
  ? normalizedApiBaseUrl.slice(0, -4)
  : normalizedApiBaseUrl;

export interface AuthStatus {
  authenticated: boolean;
  token_info: {
    api_key: string;
    created_at: string;
    expires_at: string | null;
  } | null;
  error?: string;
}

export interface LoginResponse {
  success: boolean;
  message: string;
  token: string | null;
}

export interface LogoutResponse {
  success: boolean;
  message: string;
}

export interface TokenResponse {
  token: string | null;
  message: string;
}

/**
 * Check current authentication status
 */
export async function checkAuthStatus(): Promise<AuthStatus> {
  const url = `${BASE_URL}/api/auth/status`;
  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Initiate OAuth login flow
 * @param updateEnv - Whether to update .env.local file
 */
export async function loginWithOAuth(updateEnv: boolean = false): Promise<LoginResponse> {
  const url = `${BASE_URL}/api/auth/login`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ update_env: updateEnv }),
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.message || `HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Logout and remove saved credentials
 */
export async function logout(): Promise<LogoutResponse> {
  const url = `${BASE_URL}/api/auth/logout`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.message || `HTTP error! status: ${response.status}`);
  }

  return response.json();
}

/**
 * Get the saved API token
 * @param unmask - Whether to return the full token (default: false for security)
 */
export async function getToken(unmask: boolean = false): Promise<TokenResponse> {
  const url = `${BASE_URL}/api/auth/token${unmask ? "?unmask=true" : ""}`;
  const response = await fetch(url, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.message || `HTTP error! status: ${response.status}`);
  }

  return response.json();
}
