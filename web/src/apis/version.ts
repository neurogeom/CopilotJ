/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { getBaseUrl } from "./base";

/**
 * Protocol version this frontend was built against.
 *
 * MUST mirror the backend `copilotj/server/protocol.py` `API_VERSION`. Only the
 * MAJOR component decides compatibility (see {@link parseApiMajor}).
 */
export const API_VERSION = "1.0";

export interface ServerVersion {
  api_version: string;
}

/** Fetch the server's protocol version via `GET /api/version`. */
export async function getServerVersion(): Promise<ServerVersion> {
  const url = `${getBaseUrl()}/version`;
  const response = await fetch(url, {
    method: "GET",
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}

/** Parse the MAJOR component of a "MAJOR.MINOR" version string. */
export function parseApiMajor(version: string | null | undefined): number | null {
  if (!version) return null;
  const match = /^(\d+)/.exec(version.trim());
  return match ? Number(match[1]) : null;
}

/** Compatible iff the server's MAJOR matches this frontend's MAJOR. */
export function isApiVersionCompatible(serverVersion: string | null | undefined): boolean {
  const server = parseApiMajor(serverVersion);
  const client = parseApiMajor(API_VERSION);
  return server !== null && client !== null && server === client;
}
