/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

import dayjs, { Dayjs } from "dayjs";
import { acceptHMRUpdate, defineStore } from "pinia";
import { computed, markRaw, reactive, ref } from "vue";
import type { ThreadConfig, ThreadConfigQuery, ToolCall, OptimizePromptResponse } from "../apis";
import {
  deleteThread,
  newThread,
  newThreadPost,
  updateThreadConfig,
  optimizePrompt as optimizePromptApi,
  optimizePromptBeforeThread,
} from "../apis";
import { useSettings } from "../store";

export type PostText = {
  id: string;
  type: "user" | "system:dialog_completed" | "error:system" | "error:assistant";
  content: { data: string; createAt: Dayjs };
  createAt: Dayjs;
};

type Action = {
  type: "action";
  content: ToolCall;
  called: boolean;
  result: { type: "success" | "error"; data: string } | null;
  createAt: Dayjs;
};

export type PostAgent = {
  id: string;
  type: "agent";
  role: string;
  thought: ({ type: "reasoning"; content: string; createAt: Dayjs } | Action)[];
  content: { data: string; createAt: Dayjs } | null;
  streaming: boolean;
  createAt: Dayjs;
};

export type Post = PostText | PostAgent;

export type ThreadStatus = "init" | "started" | "error" | "aborted";

// TODO: merge into useActiveThread?
export function useThread(): {
  onUpdate(cb: () => void): void;
  updateConfig(newConfig: ThreadConfigQuery): Promise<void>;
  send(content: string): Promise<void>;
  optimizePrompt(prompt: string): Promise<string | null>;
  abort(): void;
  readonly config: ThreadConfig | null;
  readonly status: ThreadStatus;
  readonly posts: Post[];
  readonly loading: boolean;
  readonly optimizing: boolean;
  readonly sendable: boolean;
} {
  const id = ref<string | null>(null);
  const status = ref<ThreadStatus>("init");
  const posts = ref<Post[]>([]);
  const loading = ref(false);
  const optimizing = ref(false);
  const abortController = ref<AbortController | null>(null);

  const sendable = computed(
    () => !loading.value && !optimizing.value && (status.value === "init" || status.value === "started"),
  );

  const config = useSettings();

  const threadConfig = ref<ThreadConfig | null>(null);

  const callbacks: (() => void)[] = [];

  function onUpdate(cb: () => void) {
    callbacks.push(cb);
  }

  async function updateConfig(newConfig: ThreadConfigQuery): Promise<void> {
    if (!id.value) return;

    const appliedConfig = await updateThreadConfig(id.value, newConfig);
    threadConfig.value = appliedConfig;
  }

  async function send(content: string) {
    if (import.meta.env.DEV) {
      console.log(status.value, ": ", content);
    }

    if (!sendable.value || !content.trim()) return;

    // Add user message to posts
    const createAt = dayjs();
    posts.value.push({
      id: generatePostId(),
      type: "user",
      content: { data: content, createAt },
      createAt,
    });
    loading.value = true;

    if (status.value === "init") {
      try {
        const appliedConfig = await newThread(config.value);
        id.value = appliedConfig.id;
        threadConfig.value = appliedConfig.config;
        if (import.meta.env.DEV) {
          console.log("New thread created with ID:", id.value);
        }
      } catch (error) {
        status.value = "error";
        loading.value = false;
        posts.value.push({
          id: generatePostId(),
          type: "error:assistant",
          content: { data: (error as any).toString(), createAt },
          createAt,
        });
        return;
      }
    } else if (status.value === "aborted") {
      throw new Error("Thread has been aborted");
    }

    // Create new abort controller for this request
    abortController.value = new AbortController();
    const stream = newThreadPost(id?.value ?? "", content, abortController.value.signal);
    status.value = "started";

    try {
      // Process streaming response
      for await (const chunk of stream) {
        if (import.meta.env.DEV) {
          console.log("Received chunk:", chunk);
        }

        switch (chunk.type) {
          case "new:post": {
            const data = chunk.data.map((a) => a.markdown).join("\n");
            posts.value.push({
              id: generatePostId(),
              type: "agent",
              role: chunk.role,
              thought: [],
              content: { data, createAt: dayjs() },
              streaming: false,
              createAt: dayjs(),
            });
            break;
          }

          case "new:post_reasoning_chunk": {
            const lastPost = ensureAgentLastPost(chunk.role);
            if (lastPost.thought.length === 0 || lastPost.thought[lastPost.thought.length - 1].type !== "reasoning") {
              lastPost.thought.push({ type: "reasoning", content: "", createAt: dayjs() });
            }
            lastPost.thought[lastPost.thought.length - 1].content += chunk.data.markdown;
            break;
          }

          case "new:post_content_chunk": {
            const lastPost = ensureAgentLastPost(chunk.role);
            if (lastPost.content) {
              lastPost.content.data += chunk.data.markdown;
            } else {
              lastPost.content = { data: chunk.data.markdown, createAt: dayjs() };
            }
            break;
          }

          case "new:tool_call": {
            const lastPost = ensureAgentLastPost(chunk.role);
            lastPost.thought.push({
              type: "action",
              content: chunk.data,
              called: false,
              result: null,
              createAt: dayjs(),
            });
            break;
          }

          case "new:tool_called": {
            const action = findAction(chunk.role, chunk.data.id);
            if (action) {
              action.called = true;
            } else {
              console.error("Tool call result not found for ID:", chunk.data.id);
            }
            break;
          }

          case "new:tool_call_result": {
            // TODO: async loading of tool call results
            const action = findAction(chunk.role, chunk.data.id);
            if (action) {
              const data = chunk.data.result.map((a) => a.markdown).join("\n");
              action.result = { type: chunk.data.type, data };
            } else {
              console.error("Tool call result not found for ID:", chunk.data.id);
            }
            break;
          }

          case "new:handoff": {
            const lastPost = ensureAgentLastPost(chunk.role);
            const at = chunk.data.name.includes(" ") ? `@"${chunk.data.name}" ` : `@${chunk.data.name} `;
            if (lastPost.content) {
              lastPost.content.data = at + lastPost.content.data + chunk.data.message;
            } else {
              lastPost.content = { data: at + chunk.data.message, createAt: dayjs() };
            }
            lastPost.streaming = false; // Mark the last post as finished
            break;
          }

          case "new:dialog_changed":
            switch (chunk.data.state) {
              case "completed":
                posts.value.push({
                  id: generatePostId(),
                  type: "system:dialog_completed",
                  content: {
                    data: chunk.data.id,
                    createAt: dayjs(),
                  },
                  createAt: dayjs(),
                });
                break;

              default:
                console.warn("Unknown dialog change state:", chunk.data.state);
            }
            break;

          case "new:error": {
            const createAt = dayjs();
            posts.value.push({
              id: generatePostId(),
              type: "error:assistant",
              content: { data: chunk.data, createAt },
              createAt,
            });
            break;
          }

          case "update:state": {
            switch (chunk.data) {
              case "agent_speaking":
                // skip if the last post is already an agent message with the same role and is streaming
                ensureAgentLastPost(chunk.role);
                break;

              case "confirmation_request":
                // TODO: add a ui component to handle confirmation requests
                posts.value.push({
                  id: generatePostId(),
                  type: "agent",
                  role: chunk.role,
                  thought: [],
                  content: {
                    data: "Confirmation required. Please respond with 'yes' or 'no'.",
                    createAt: dayjs(),
                  },
                  streaming: false,
                  createAt: dayjs(),
                });
                break;

              case "agent_finish":
                if (posts.value.length > 0) {
                  const lastPost = posts.value[posts.value.length - 1];
                  if (lastPost.type === "agent" && lastPost.streaming) {
                    lastPost.streaming = false; // Mark the last post as finished
                  }
                }
                break;

              default:
                // To be safe, let's cast chunk to any to check its type property
                console.warn("Unknown message:", chunk);
                break;
            }

            break;
          }

          default:
            // To be safe, let's cast chunk to any to check its type property
            console.warn("Unknown message:", chunk);
            break;
        }

        for (const cb of callbacks) {
          cb();
        }
      }

      if (posts.value.length > 1) {
        const lastPost = posts.value[posts.value.length - 1];
        if (lastPost.type === "agent" && lastPost.streaming) {
          // TODO: Handle unexpected end of stream gracefully
          console.error("Unexpected end of stream, marking last post as finished");
          lastPost.streaming = false; // Mark the last post as finished
          status.value = "error"; // We dont expect to recover from this
        }
      } else {
        // only user post exists, something went wrong
        const createAt = dayjs();
        posts.value.push({
          id: generatePostId(),
          type: "error:assistant",
          content: { data: "Sorry, there was an error processing your request.", createAt },
          createAt,
        });

        console.error("No posts received in the thread");
        status.value = "error"; // We dont expect to recover from this
      }

      if (import.meta.env.DEV) {
        console.log("End of chunk processing");
      }
    } catch (error) {
      // Handle specific error types
      if (error instanceof Error && error.name === "AbortError") {
        console.log("Thread request was aborted");
        status.value = "aborted";
      } else {
        // TODO: Handle specific error types if needed, e.g: resend message in the next tick
        console.error("Error in thread:", error);
        const createAt = dayjs();
        posts.value.push({
          id: generatePostId(),
          type: "error:system",
          content: { data: "Sorry, there was an error processing your request.", createAt },
          createAt,
        });
        status.value = "error"; // we dont expect to recover from this
      }
    } finally {
      abortController.value = null;
      loading.value = false;
    }
  }

  function ensureAgentLastPost(role: string): PostAgent {
    if (posts.value.length !== 0) {
      const lastPost = posts.value[posts.value.length - 1];
      if (lastPost.type === "agent" && lastPost.role === role) {
        lastPost.streaming = true; // Mark the last post as streaming
        return lastPost as PostAgent; // Return the last post if it matches the role
      }
    }

    const lastPost: PostAgent = {
      id: generatePostId(),
      type: "agent",
      role: role,
      thought: [],
      content: null,
      streaming: true,
      createAt: dayjs(),
    };
    posts.value.push(lastPost);
    return lastPost;
  }

  function findAction(role: string, toolCallId: string): Action | null {
    // NOTE: dont assume that the tool call result is always present in the last post
    for (const post of posts.value) {
      if (post.type !== "agent" || post.role !== role) continue;

      for (const item of post.thought) {
        if (item.type !== "action" || item.content.id !== toolCallId) continue;

        if (item.result !== null) {
          console.error("Unexpected tool call result, already has a result:", item);
        } else {
          return item;
        }
      }
    }
    return null;
  }

  function abort() {
    // Abort ongoing request
    if (abortController.value) {
      abortController.value.abort();
      abortController.value = null;
    }

    // Mark the last post as finished if it's an agent post
    if (posts.value.length > 0) {
      const lastPost = posts.value[posts.value.length - 1];
      if (lastPost.type === "agent" && lastPost.streaming) {
        lastPost.streaming = false; // Mark the last post as finished
      }
    }

    // Update status
    status.value = "aborted";
    loading.value = false;

    for (const cb of callbacks) {
      cb();
    }

    // Only delete when a thread was actually created; reset() in the init state
    // (e.g. the "New Thread" button with no thread yet) leaves id null.
    if (id.value) deleteThread(id.value); // no await
  }

  async function optimizePrompt(prompt: string): Promise<string | null> {
    // Prevent concurrent optimization requests
    if (optimizing.value) {
      return null;
    }

    optimizing.value = true;

    try {
      let result: OptimizePromptResponse;

      // Use standalone endpoint if thread doesn't exist yet
      if (status.value === "init" || !id.value) {
        result = await optimizePromptBeforeThread(prompt);
      } else {
        result = await optimizePromptApi(id.value, prompt);
      }

      return result.optimized;
    } catch (error) {
      console.error("Failed to optimize prompt:", error);
      return null;
    } finally {
      optimizing.value = false;
    }
  }

  return reactive({
    onUpdate: markRaw(onUpdate),
    updateConfig: markRaw(updateConfig),
    send: markRaw(send),
    optimizePrompt: markRaw(optimizePrompt),
    abort: markRaw(abort),
    config: threadConfig,
    status,
    posts,
    loading,
    optimizing,
    sendable,
  });
}

export const useActiveThread = defineStore("active-thread", () => {
  const thread = ref(useThread());

  function reset() {
    thread.value.abort();
    thread.value = useThread();
  }

  return { thread, reset };
});

// Simple ID generator for posts
// TODO: use id from backend
function generatePostId() {
  return `post_${generateUUID()}`;
}

// NOTE: Simple UUID generator, not cryptographically secure
function generateUUID() {
  // Public Domain/MIT
  var d = new Date().getTime(); //Timestamp
  var d2 = (typeof performance !== "undefined" && performance.now && performance.now() * 1000) || 0; //Time in microseconds since page-load or 0 if unsupported
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    var r = Math.random() * 16; //random number between 0 and 16
    if (d > 0) {
      //Use timestamp until depleted
      r = ((d + r) % 16) | 0;
      d = Math.floor(d / 16);
    } else {
      //Use microseconds since page-load if supported
      r = ((d2 + r) % 16) | 0;
      d2 = Math.floor(d2 / 16);
    }
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useActiveThread, import.meta.hot));
}
