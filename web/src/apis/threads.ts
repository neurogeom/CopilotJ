/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { baseUrl } from "./base.ts";

interface Payload<T extends string, K> {
  type: T;
  role: string;
  data: K;
}

type ContentMarkdown = {
  type: "markdown";
  markdown: string;
};

type Content = ContentMarkdown;

export type StateMessage = Payload<"update:state", "agent_speaking" | "agent_finish" | "confirmation_request">;

export type PostMessage = Payload<"new:post", Content[]>;

export type PostReasoningChunkMessage = Payload<"new:post_reasoning_chunk", Content>;

export type PostContentChunkMessage = Payload<"new:post_content_chunk", Content>;

export type ErrorMessage = Payload<"new:error", string>;

export interface ToolCall {
  id: string;
  tool: {
    name: string;
    display_name: string;
    description: string;
  };
  args: Record<string, any>;
}

export type ToolCallMessage = Payload<"new:tool_call", ToolCall>;

export type ToolCallArgsMessage = Payload<"new:tool_called", { id: string }>;

export type ToolCallResultMessage = Payload<
  "new:tool_call_result",
  { id: string; type: "success" | "error"; result: Content[] }
>;

export type HandoffMessage = Payload<"new:handoff", { id: string; name: string; message: string }>;

export type DialogChangeMessage = Payload<"new:dialog_changed", { id: string; state: "completed" }>;

// Union of all possible messages from the stream, except for the initial thread ID
export type TypedMessage =
  | StateMessage
  | PostMessage
  | PostReasoningChunkMessage
  | PostContentChunkMessage
  | ErrorMessage
  | ToolCallMessage
  | ToolCallArgsMessage
  | ToolCallResultMessage
  | HandoffMessage
  | DialogChangeMessage;

export type NewThread = {
  id: string;
  config: ThreadConfig;
};

export interface ThreadConfigModel {
  name: string;
  api_key: string | null;
  base_url: string | null;
}

export interface ThreadConfig {
  model: ThreadConfigModel;
}

export interface ThreadConfigQuery {
  model?: ThreadConfigModel | null;
}

export interface OptimizePromptRequest {
  prompt: string;
}

export interface OptimizePromptResponse {
  original: string;
  optimized: string;
}

export async function newThread(config: ThreadConfigQuery): Promise<NewThread> {
  const url = `${baseUrl}/threads`;
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  };
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function* newThreadPost(
  threadId: string,
  prompt: string | boolean,
  signal?: AbortSignal,
): AsyncGenerator<TypedMessage> {
  const url = `${baseUrl}/threads/${threadId}/posts`;
  const options: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
    signal,
  };
  yield* ndjsonStream<TypedMessage>(url, options);
}

export async function getThreadConfig(threadId: string): Promise<ThreadConfig> {
  const url = `${baseUrl}/threads/${threadId}/config`;
  const options = { method: "GET", headers: { "Content-Type": "application/json" } };

  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function updateThreadConfig(threadId: string, config: ThreadConfigQuery): Promise<ThreadConfig> {
  const url = `${baseUrl}/threads/${threadId}/config`;
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  };

  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function deleteThread(threadId: string): Promise<void> {
  const url = `${baseUrl}/threads/${threadId}`;
  const options = { method: "DELETE" };
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function optimizePrompt(threadId: string, prompt: string): Promise<OptimizePromptResponse> {
  const url = `${baseUrl}/threads/${threadId}/optimize-prompt`;
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  };
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function optimizePromptBeforeThread(prompt: string): Promise<OptimizePromptResponse> {
  const url = `${baseUrl}/optimize-prompt`;
  const options = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  };
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

async function* ndjsonStream<T>(url: string, options?: RequestInit): AsyncGenerator<T> {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
  if (!response.body) throw new Error("Response body is null");

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");

  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.trim() === "") continue;
      try {
        yield JSON.parse(line) as T;
      } catch (err) {
        console.error("Failed to parse JSON:", err, "Line:", line);
      }
    }
  }

  if (buffer.trim() !== "") {
    try {
      yield JSON.parse(buffer) as T;
    } catch (err) {
      console.error("Failed to parse final JSON:", err, "Buffer:", buffer);
    }
  }
}
