/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

export function getModelName(modelId: string): string {
  switch (modelId) {
    // Anthropic
    case "claude-opus-4-6":
      return "Claude Opus 4.6";
    case "claude-sonnet-4-6":
      return "Claude Sonnet 4.6";
    case "claude-haiku-4-5-20251001":
      return "Claude Haiku 4.5";

    // OpenAI
    case "gpt-5.4":
      return "GPT-5.4";
    case "gpt-5.4-mini":
      return "GPT-5.4 mini";
    case "gpt-5":
      return "GPT-5";
    case "gpt-5-mini":
      return "GPT-5 mini";
    case "gpt-5-nano":
      return "GPT-5 nano";
    case "gpt-4.1":
      return "GPT-4.1";
    case "gpt-4.1-mini":
      return "GPT-4.1 mini";
    case "gpt-4.1-nano":
      return "GPT-4.1 nano";
    case "gpt-4o":
      return "GPT-4o";
    case "gpt-4o-mini":
      return "GPT-4o mini";
    case "gpt-4":
      return "GPT-4";
    case "gpt-4-mini":
      return "GPT-4 mini";
    case "gpt-4-nano":
      return "GPT-4 nano";
    case "gpt-3.5":
      return "GPT-3.5";

    // Google
    case "gemini-3.1-pro-preview":
      return "Gemini 3.1 Pro";
    case "gemini-3-flash-preview":
      return "Gemini 3 Flash";
    case "gemini-3.1-flash-lite-preview":
      return "Gemini 3.1 Flash Lite";
    case "gemini-2.5-pro":
      return "Gemini 2.5 Pro";
    case "gemini-2.5-flash":
      return "Gemini 2.5 Flash";
    case "gemini-2.5-flash-lite":
      return "Gemini 2.5 Flash Lite";

    default:
      // For ollama/ prefix, strip the prefix for display
      if (modelId.startsWith("ollama/")) return modelId.slice(7);
      return modelId;
  }
}
