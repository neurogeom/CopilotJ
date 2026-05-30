/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.awt.Component;
import java.awt.Panel;
import java.awt.Window;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.scijava.Context;
import org.scijava.log.LogService;
import org.scijava.script.ScriptModule;
import org.scijava.script.ScriptService;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import ij.IJ;
import ij.gui.MessageDialog;
import ij.gui.MultiLineLabel;
import ij.macro.Interpreter;

class ScriptRunner {
  public static class Result {
    public final Map<String, Object> outputs;
    public final String err;

    public Result(final Map<String, Object> outputs) {
      this.outputs = outputs;
      this.err = null;
    }

    public Result(final String err) {
      this.outputs = null;
      this.err = err;
    }
  }

  public static class PluginRequest {
    public String plugin;
    public ObjectNode args;
    public boolean ij1Style = true;
  }

  public static class ScriptRequest {
    public String language;
    public String script;
  }

  // Callback interface for async execution
  public interface ScriptCallback {
    void onComplete(Result result);
  }

  private final LogService log;
  private final ScriptService scriptService;

  public ScriptRunner(final Context context, final LogService log) {
    this.log = log;
    this.scriptService = context.service(ScriptService.class);
  }

  public Result runPlugin(final PluginRequest request) {
    // check
    if (request.plugin == null || request.plugin == "") {
      return new Result("Plugin name is required");
    }

    final String argline = formatArguments(request.args, request.ij1Style);

    IJ.run(request.plugin, argline); // TODO: capture error
    return new Result(new HashMap<>());
  }

  public Result runScript(final ScriptRequest request) {
    // Check
    if (scriptService == null) {
      return new Result("Script service not available");
    } else if (request == null) {
      return new Result("Script request is null");
    } else if (request.language == null || request.language == "") {
      return new Result("Language is required");
    } else if (request.script == null || request.script == "") {
      return new Result("Script is required");
    }

    // Check if a blocking window is already open
    final String blockingWindowMessage = checkForBlockingWindow();
    if (blockingWindowMessage != null) {
      return new Result("Cannot run script: " + blockingWindowMessage);
    }

    // Determine script language
    final String extension = getLanguageExtension(request.language);

    // Use a future-like pattern to wait for the async result
    final Result[] resultHolder = new Result[1];
    final boolean[] completed = new boolean[1];

    final ScriptCallback callback = new ScriptCallback() {
      @Override
      public void onComplete(Result result) {
        synchronized (completed) {
          if (completed[0]) {
            return; // Already completed
          }

          resultHolder[0] = result;
          completed[0] = true;
          completed.notifyAll(); // Wake up waiting thread
        }
      }
    };

    // Create and start the dialog killer thread
    final Thread modalMonitor = new Thread(() -> runModalMonitor(500, callback));

    // Create the script in a separate thread
    final Thread scriptThread = new Thread(() -> {
      try {
        // TODO: check if scripting header not exists
        final boolean isIj1Macro = "ijm".equals(extension);

        final Result result = isIj1Macro
            ? runIJ1Macro(request.script)
            : runIJ2Script(request.script, extension);

        if (callback != null) {
          callback.onComplete(result);
        }
      } catch (Exception e) {
        if (callback != null) {
          callback.onComplete(new Result("Exception in script execution: " + e.getMessage()));
        }
      }
    });

    synchronized (completed) {
      try {
        modalMonitor.setDaemon(true);
        modalMonitor.start();

        scriptThread.setDaemon(true);
        scriptThread.start();

        // Wait for completion
        while (!completed[0]) {
          completed.wait(); // Wait until notified
        }

      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return new Result("Script execution interrupted");

      } finally {
        // Stop the all threads
        modalMonitor.interrupt();
        scriptThread.interrupt();

        try {
          modalMonitor.join(1000); // Wait for up to 1 second
        } catch (InterruptedException e) {
          log.warn("Modal monitor interrupted while stopping");
        }

        try {
          scriptThread.join(1000); // Wait for up to 1 second
        } catch (InterruptedException e) {
          log.warn("Script thread interrupted while stopping");
        }
      }
    }

    return resultHolder[0];
  }

  private Result runIJ1Macro(String script) {
    script = rewriteIj1Macro(script);

    final Interpreter interpreter = new Interpreter();
    try {
      final String result = interpreter.eval(script);
      if ((result == null || result.isEmpty())) {
        return new Result("No output from macro");
      } else if (result != null && result.toLowerCase().startsWith("error:")) {
        return new Result(result.substring(6).trim());
      }

      final Map<String, Object> outputs = new HashMap<>();
      outputs.put("result", result);
      return new Result(outputs);

    } catch (final StringIndexOutOfBoundsException e) {
      final String err = e.getMessage() + "\n Tip: likely due to syntax error";
      return new Result(err);

    } catch (final RuntimeException e) {
      String err = interpreter.getErrorMessage();
      if (err == null || err == "") {
        err = e.getMessage();
      }
      return new Result(err);

    } catch (final Exception e) {
      return new Result(e.getMessage());
    }
  }

  private Result runIJ2Script(final String script, final String extension) {
    // Run ImageJ2 Script
    try {
      final ScriptModule scriptModule = scriptService.run("script." + extension, script, true).get();
      final Map<String, Object> outputs = scriptModule.getOutputs();
      if ((outputs == null || outputs.isEmpty())) {
        return new Result("No output from script.");
      }
      return new Result(outputs);

    } catch (final IllegalArgumentException e) {
      return new Result(e.getMessage());

    } catch (final ExecutionException e) {
      log.error("Error executing `" + extension + "` script:\n" + script);

      // Unwrap exceptions
      Throwable err = e; // ExecutionException

      // RuntimeException
      err = (err.getCause() != null) ? err.getCause() : err;
      String message;
      switch (err.getMessage()) {
        // https://github.com/scijava/scijava-common/blob/7ac4926d72f3ce3051787629a021d5f3f3adfdf2/src/main/java/org/scijava/module/ModuleRunner.java#L123-L134
        case "Module threw exception":
          message = "Script threw exception: ";
          break;

        case "Module threw error":
          message = "Script threw error: ";
          break;

        default:
          message = "Script execution error: ";
          break;
      }

      // The module exception
      err = (err.getCause() != null) ? err.getCause() : err;

      // Detect error message
      if (err instanceof NullPointerException) {
        message += "Null pointer";
      } else {
        message += err.getMessage();
      }

      return new Result(message);

    } catch (final Exception e) {
      e.printStackTrace();
      return new Result("Unexpected error: " + e.getMessage());
    }
  }

  /**
   * Check if there's already a blocking/modal window open
   *
   * @return A message describing the blocking window, or null if none found
   */
  private String checkForBlockingWindow() {
    for (Window w : Window.getWindows()) {
      if (!w.isVisible()) {
        continue;
      }

      if (w instanceof java.awt.Dialog) {
        final java.awt.Dialog d = (java.awt.Dialog) w;
        if (d.isModal()) {
          return "Modal dialog is already open: " + (d.getTitle() != null ? d.getTitle() : "Untitled");
        }
      } else if (w instanceof javax.swing.JDialog) {
        final javax.swing.JDialog d = (javax.swing.JDialog) w;
        if (d.isModal()) {
          return "Modal dialog is already open: " + (d.getTitle() != null ? d.getTitle() : "Untitled");
        }
      }
    }
    return null; // No blocking window found
  }

  /**
   * Modal window monitor to close blocking dialogs
   */

  // Simplified modal monitor as a method
  private void runModalMonitor(final int pollIntervalMillis, ScriptCallback callback) {
    log.debug("[ModalMonitor] Starting modal dialog killer thread with interval: " + pollIntervalMillis + " ms");
    while (true) {
      try {
        for (Window w : Window.getWindows()) {
          if (!w.isVisible()) {
            continue;
          }

          if (w instanceof java.awt.Dialog) {
            final java.awt.Dialog d = (java.awt.Dialog) w;
            if (!d.isModal()) {
              continue;
            }
            log.info("[ModalMonitor] Detect modal dialog: " + w.getName() + " " + d.getTitle());

            if (d instanceof MessageDialog) {
              final MessageDialog md = (MessageDialog) d;
              final String title = md.getTitle();
              if (title == null) {
                continue;
              }

              String message = "<unknown>";
              for (Component child : md.getComponents()) {
                log.debug("[ModalMonitor] " + child.getClass().getName() + ", " + child.getName());

                if (child instanceof Panel) {
                  for (Component panelChild : ((Panel) child).getComponents()) {
                    if (panelChild instanceof MultiLineLabel) {
                      try {
                        final Field linesField = panelChild.getClass().getDeclaredField("lines");
                        linesField.setAccessible(true);
                        final String[] lines = (String[]) linesField.get(panelChild);
                        message = String.join(". ", lines);
                      } catch (Exception e) {
                        log.info("[ModalMonitor] Error retrieving dialog message: " + e.getMessage());
                      }
                      break;
                    }
                  }
                }
              }

              log.info("[ModalMonitor] Closing dialog: " + title + " - " + message);
              callback.onComplete(new Result(message)); // NOTE: Call callback before closing the dialog
              md.setVisible(false);
              md.actionPerformed(null); // Close the dialog
              break;

            } else {
              final String message = "Script aborted due to modal dialog titled: " + d.getTitle()
                  + ". Please ask user to check and close it.";
              callback.onComplete(new Result(message));
            }

            break; // Exit the loop after closing the dialog

          } else if (w instanceof javax.swing.JDialog) {
            final javax.swing.JDialog d = (javax.swing.JDialog) w;
            if (!d.isModal()) {
              continue;
            }

            log.error("Modal dialog: " + w.getName() + " " + d.getTitle());
            final String message = "Script aborted due to modal dialog titled: " + d.getTitle()
                + ". Please ask user to check and close it.";
            callback.onComplete(new Result(message));
            break; // Exit the loop after closing the dialog
          }
        }

        Thread.sleep(pollIntervalMillis);
      } catch (InterruptedException e) {
        log.debug("[ModalMonitor] Modal dialog killer thread interrupted, stopping...");
        break;
      } catch (Exception e) {
        log.error("[ModalMonitor] Unexpected error: " + e.getMessage());
        // Continue running unless explicitly stopped
      }
    }
  }

  /**
   * Arguments
   */

  private String formatArguments(final ObjectNode args, final boolean ij1Style) {
    if (args == null || args.isNull()) {
      return "";

    } else if (args.isTextual()) {
      return args.asText();

    } else if (args.isObject()) {
      final List<String> formattedArgs = new ArrayList<>();
      final Iterator<Map.Entry<String, JsonNode>> it = args.fields();
      while (it.hasNext()) {
        final Map.Entry<String, JsonNode> entry = it.next();
        formattedArgs.add(formatArgument(entry.getKey(), entry.getValue(), ij1Style));
      }
      return String.join(" ", formattedArgs);
    }

    return "";
  }

  private String formatArgument(final String key, final JsonNode value, final boolean ij1Style) {
    if (value.isBoolean()) {
      if (value.asBoolean()) {
        return ij1Style ? key : key + "=true";
      } else {
        return ij1Style ? "" : key + "=false";
      }

    } else if (value.isTextual()) {
      final String v = value.asText().replace("\\", "/");
      if (v.startsWith("[") && v.endsWith("]")) {
        return key + "=" + v;
      }
      return key + "=[" + v + "]";

    } else if (value.isNumber()) {
      return key + "=[" + value.asDouble() + "]";

    } else if (value.isNull()) {
      return ""; // Convert for None is not yet implemented

    } else if (value.isObject()) {
      throw new IllegalArgumentException("Argument " + key + " is Object type which is not valid");

    } else {
      throw new IllegalArgumentException("Argument " + key + " has unsupported type: " + value.getNodeType());
    }
  }

  private String getLanguageExtension(final String language) {
    switch (language.toLowerCase()) {
      case "macro":
      case "ijm":
        return "ijm";

      case "java":
      case "javac":
        return "java";

      case "jython":
      case "python":
      case "py":
        return "py";

      case "javascript":
      case "js":
        return "js";

      default:
        throw new IllegalArgumentException("Unsupported language: " + language +
            ". Supported languages: macro, java, jython, javascript");
    }
  }

  /**
   * IJ1 Macro Rewriting
   *
   * Rewrites certain IJ1 macro commands to prevent blocking pop-ups
   * such as file-not-found dialogs or exit dialogs.
   */

  private String rewriteIj1Macro(String macroScript) {
    macroScript = rewriteIj1MacroOpen(macroScript);
    macroScript = rewriteIj1MacroExit(macroScript);
    return macroScript;
  }

  private String rewriteIj1MacroOpen(String macroScript) {
    // Define the safeOpen(path) function that prevents file-not-found pop-ups
    final String safeOpenDef = ""
        + "function safeOpen(path) {\n"
        + "  if (File.exists(path)) {\n"
        + "    open(path);\n"
        + "  } else {\n"
        + "    print(\"Error: File not found - \" + path);\n"
        + "    return;\n"
        + "  }\n"
        + "}\n";

    // Use a pattern to match all open(...) calls
    final String pattern = "\\bopen\\s*\\(([^\\)]+)\\)";

    // Check whether the macro script contains open(...) to be replaced
    if (macroScript.matches("(?s).*" + pattern + ".*")) {
      // Replace all open(...) with safeOpen(...)
      macroScript = macroScript.replaceAll(pattern, "safeOpen($1)");

      // Inject safeOpenDef only if not already present
      if (!macroScript.contains("function safeOpen(")) {
        macroScript = safeOpenDef + "\n" + macroScript;
      }
    }

    return macroScript;
  }

  // Replace all exit(...) calls with print(...) + return to suppress GUI pop-ups
  // Preserves the original message content
  private String rewriteIj1MacroExit(final String macroScript) {
    return macroScript.replaceAll(
        "\\bexit\\s*\\(([^\\)]+)\\)",
        "print(\"Error: \" + $1); return;");
  }
}
