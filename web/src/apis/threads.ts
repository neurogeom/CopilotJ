/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import { getBaseUrl } from "./base";

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

export type RetryMessage = Payload<
  "update:retry",
  { attempt: number; max_attempts: number; wait_seconds: number; reason: string }
>;

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
  | RetryMessage
  | ToolCallMessage
  | ToolCallArgsMessage
  | ToolCallResultMessage
  | HandoffMessage
  | DialogChangeMessage;

export type NewThread = {
  id: string;
  config: ThreadConfig;
};

/** An explicit, user-configured model. Also the resolved shape returned by the server. */
export interface ExplicitModel {
  name: string;
  api_key: string | null;
  base_url: string | null;
  provider: string | null;
}

/** "Use the server's env-configured model" — sent instead of an ExplicitModel. */
export interface UseServerModel {
  use_server: true;
}

/** A model in a config query: either an explicit model or "use the server's". */
export type ThreadConfigModel = UseServerModel | ExplicitModel;

export function isUseServer(m: ThreadConfigModel | null | undefined): m is UseServerModel {
  return m != null && (m as UseServerModel).use_server === true;
}

export function isExplicit(m: ThreadConfigModel | null | undefined): m is ExplicitModel {
  return m != null && !isUseServer(m);
}

export interface ThreadConfig {
  model: ExplicitModel;
}

export interface ThreadConfigQuery {
  model?: ThreadConfigModel | null;
  vlm?: ThreadConfigModel | null;
  vision_enabled?: boolean | null;
  proxy?: string | null;
  tavily_api_key?: string | null;
  kb_autosave?: boolean;
}

export interface OptimizePromptRequest {
  prompt: string;
}

export interface OptimizePromptResponse {
  original: string;
  optimized: string;
}

export interface ServerConfig {
  model: ExplicitModel | null;
  vlm: ExplicitModel | null;
  proxy: string | null;
  kb_autosave: boolean;
  vision_enabled: boolean;
  llm_supports_vision: boolean;
}

export async function getServerConfig(): Promise<ServerConfig> {
  const url = `${getBaseUrl()}/config`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export interface ModelCapabilities {
  model: string;
  supports_vision: boolean;
  supports_function_calling: boolean;
  source: string;
}

export async function getModelCapabilities(model: string): Promise<ModelCapabilities> {
  const url = `${getBaseUrl()}/model/capabilities?model=${encodeURIComponent(model)}`;
  const response = await fetch(url, { method: "GET" });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function newThread(config: ThreadConfigQuery): Promise<NewThread> {
  const url = `${getBaseUrl()}/threads`;
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
  const url = `${getBaseUrl()}/threads/${threadId}/posts`;
  const options: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
    signal,
  };
  yield* ndjsonStream<TypedMessage>(url, options);
}

export async function getThreadConfig(threadId: string): Promise<ThreadConfig> {
  const url = `${getBaseUrl()}/threads/${threadId}/config`;
  const options = { method: "GET", headers: { "Content-Type": "application/json" } };

  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function updateThreadConfig(threadId: string, config: ThreadConfigQuery): Promise<ThreadConfig> {
  const url = `${getBaseUrl()}/threads/${threadId}/config`;
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
  const url = `${getBaseUrl()}/threads/${threadId}`;
  const options = { method: "DELETE" };
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
  }
  return response.json();
}

export async function optimizePrompt(threadId: string, prompt: string): Promise<OptimizePromptResponse> {
  const url = `${getBaseUrl()}/threads/${threadId}/optimize-prompt`;
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
  const url = `${getBaseUrl()}/optimize-prompt`;
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
