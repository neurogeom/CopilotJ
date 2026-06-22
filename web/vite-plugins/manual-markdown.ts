/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFile } from "node:fs/promises";
import type { Plugin } from "vite";
import { createMarkdownRenderer } from "../src/lib/markdown";

type TocItem = { level: number; text: string; id: string };

// Slug-ification migrated verbatim from Manual.vue so generated IDs stay identical.
function slugify(input: string, counts: Map<string, number>) {
  let s = input
    .toLowerCase()
    .trim()
    .replace(/[`*_~]/g, "")
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-");
  if (!s) s = "section";
  const n = counts.get(s) || 0;
  counts.set(s, n + 1);
  return n ? `${s}-${n}` : s;
}

function getHeadingText(token: any): string {
  if (token.type === "text") return token.text;
  if (token.tokens) {
    return token.tokens.map(getHeadingText).join(" ");
  }
  return "";
}

// Renders the manual markdown into HTML (with header IDs + syntax highlighting) and
// builds the table of contents, all at build time. The browser only consumes the result.
function buildManual(src: string): { html: string; toc: TocItem[] } {
  const headingCounts = new Map<string, number>();
  const renderer = {
    heading({ tokens, depth }: { tokens: any[]; depth: number }) {
      const text = tokens.map(getHeadingText).join(" ");
      const id = slugify(text, headingCounts);
      const tag = `h${depth}`;
      return `<${tag} id="${id}">${text}</${tag}>\n`;
    },
  };

  const markdown = createMarkdownRenderer(renderer);
  const html = markdown.parse(src) as string;

  const tocCounts = new Map<string, number>();
  const toc: TocItem[] = [];
  for (const token of markdown.lexer(src)) {
    if (token.type === "heading" && (token.depth === 2 || token.depth === 3)) {
      const text = token.tokens?.map(getHeadingText).join(" ") || "";
      const id = slugify(text, tocCounts);
      toc.push({ level: token.depth, text, id });
    }
  }

  return { html, toc };
}

// Loads `src/assets/manual.md` as a JS module exporting { html, toc }. Handling the
// file itself (rather than a `?query`) keeps it directly file-linked in Vite's module
// graph, so edits hot-update in place instead of triggering a full page reload.
export function manualMarkdownPlugin(): Plugin {
  return {
    name: "copilotj-manual-markdown",
    enforce: "pre",
    async load(id) {
      if (!id.replace(/\\/g, "/").endsWith("/assets/manual.md")) return;
      const src = await readFile(id, "utf8");
      const { html, toc } = buildManual(src);
      return {
        code: `export const html = ${JSON.stringify(html)};\nexport const toc = ${JSON.stringify(toc)};\n`,
        map: null,
      };
    },
  };
}
