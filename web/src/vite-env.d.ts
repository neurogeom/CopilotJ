/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="vite/client" />

declare module "*/assets/manual.md" {
  export interface ManualTocItem {
    level: number;
    text: string;
    id: string;
  }
  export interface ManualTab {
    id: string;
    name: string;
    html: string;
  }
  export type ManualBlock = { type: "html"; html: string } | { type: "tabs"; tabs: ManualTab[] };
  export const blocks: ManualBlock[];
  export const toc: ManualTocItem[];
}

interface ImportMetaEnv {
  readonly VITE_DEFAULT_ROUTE?: string;
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_CONFIGURABLE_API_BASE?: boolean;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
