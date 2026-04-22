/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { baseUrl } from "./base.ts";

export interface ClientInfo {
  id: string;
}

export async function listClients(): Promise<ClientInfo[]> {
  const url = `${baseUrl}/clients`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}
