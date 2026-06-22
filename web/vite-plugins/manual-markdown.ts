/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFile } from "node:fs/promises";
import type { Plugin } from "vite";
import { createMarkdownRenderer } from "../src/lib/markdown";

type TocItem = { level: number; text: string; id: string };
type ManualTab = { id: string; name: string; html: string };
type ManualBlock = { type: "html"; html: string } | { type: "tabs"; tabs: ManualTab[] };

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

// Renders the manual into ordered { blocks, toc } at build time:
//   - normal markdown  -> { type: "html", html }
//   - `::: tabs` fenced regions -> { type: "tabs", tabs: [{ id, name, html }] }
//     where each tab starts with `=== Name` and its body is markdown.
// `:::` is a half-standard container fence (MkDocs Material / vuepress). FAQ entries
// are authored directly as `<details>` in the markdown (see manual.md) and need no
// special handling here — marked passes them through and renders their inner markdown.
function buildManual(src: string): { blocks: ManualBlock[]; toc: TocItem[] } {
  const headingCounts = new Map<string, number>();
  const renderer = {
    heading({ tokens, depth }: { tokens: any[]; depth: number }) {
      const text = tokens.map(getHeadingText).join(" ");
      const id = slugify(text, headingCounts);
      return `<h${depth} id="${id}">${text}</h${depth}>\n`;
    },
  };
  const markdown = createMarkdownRenderer(renderer);

  type Segment = { kind: "prose"; text: string } | { kind: "tabs"; tabs: { name: string; content: string }[] };
  const lines = src.split("\n");
  const segments: Segment[] = [];
  let prose: string[] = [];
  const flushProse = () => {
    if (prose.length) {
      segments.push({ kind: "prose", text: prose.join("\n") });
      prose = [];
    }
  };

  // Line-scan: split into ordered prose segments and ::: tabs blocks.
  let i = 0;
  while (i < lines.length) {
    if (/^:::\s*tabs\s*$/.test(lines[i]!)) {
      flushProse();
      i++;
      const tabs: { name: string; content: string }[] = [];
      let current: { name: string; content: string } | null = null;
      while (i < lines.length && !/^:::\s*$/.test(lines[i]!)) {
        const m = /^===\s+(.+?)\s*$/.exec(lines[i]!);
        if (m) {
          if (current) tabs.push(current);
          current = { name: m[1]!, content: "" };
        } else if (current) {
          current.content += (current.content ? "\n" : "") + lines[i];
        }
        i++;
      }
      if (current) tabs.push(current);
      i++; // skip closing :::
      segments.push({ kind: "tabs", tabs });
    } else {
      prose.push(lines[i]!);
      i++;
    }
  }
  flushProse();

  // Render segments to blocks (prose and tab bodies via the same marked pipeline).
  const blocks: ManualBlock[] = [];
  for (const seg of segments) {
    if (seg.kind === "prose") {
      const html = markdown.parse(seg.text) as string;
      if (html.trim()) blocks.push({ type: "html", html });
    } else {
      const tabs: ManualTab[] = seg.tabs.map((t) => ({
        id: slugify(t.name, headingCounts),
        name: t.name,
        html: markdown.parse(t.content) as string,
      }));
      blocks.push({ type: "tabs", tabs });
    }
  }

  // TOC: depth-2/3 headings from prose segments (tab bodies and <details> excluded).
  const tocCounts = new Map<string, number>();
  const toc: TocItem[] = [];
  for (const seg of segments) {
    if (seg.kind !== "prose") continue;
    for (const token of markdown.lexer(seg.text)) {
      if (token.type === "heading" && (token.depth === 2 || token.depth === 3)) {
        const text = token.tokens?.map(getHeadingText).join(" ") || "";
        const id = slugify(text, tocCounts);
        toc.push({ level: token.depth, text, id });
      }
    }
  }

  return { blocks, toc };
}

// Loads `src/assets/manual.md` as a JS module exporting { blocks, toc }. Handling the
// file itself (rather than a `?query`) keeps it directly file-linked in Vite's module
// graph, so edits hot-update in place instead of triggering a full page reload.
export function manualMarkdownPlugin(): Plugin {
  return {
    name: "copilotj-manual-markdown",
    enforce: "pre",
    async load(id) {
      if (!id.replace(/\\/g, "/").endsWith("/assets/manual.md")) return;
      const src = await readFile(id, "utf8");
      const { blocks, toc } = buildManual(src);
      return {
        code: `export const blocks = ${JSON.stringify(blocks)};\nexport const toc = ${JSON.stringify(toc)};\n`,
        map: null,
      };
    },
  };
}
