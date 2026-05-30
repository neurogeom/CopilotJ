/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import java.awt.Dimension;
import java.awt.Window;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.scijava.log.LogService;

import ij.IJ;
import ij.Prefs;
import ij.WindowManager;
import ij.gui.ImageWindow;
import copilotj.ImagejListener;
import copilotj.awt.window.AbstractAwtWindow;
import copilotj.awt.window.AwtWindow;
import copilotj.awt.window.AwtWindowProvider;
import copilotj.util.FromTo;

public class Snapshot {

  public static class Difference {

    public static class WindowsDifference {

      public static class AwtWindowAndDifference {
        public final AwtWindow later;
        public final AbstractAwtWindow.Difference difference;

        public AwtWindowAndDifference(final AwtWindow later, final AwtWindow.Difference difference) {
          this.later = later;
          this.difference = difference;
        }
      }

      public final List<AwtWindow> added = new ArrayList<>();
      public final List<AwtWindowAndDifference> changed = new ArrayList<>();
      public final List<AwtWindow> removed = new ArrayList<>();
      public final List<AwtWindow> unchanged = new ArrayList<>();

      public WindowsDifference(final List<AwtWindow> earlyList, final List<AwtWindow> laterList,
          final ImagejListener.HistoryResponse history) {
        final Map<Integer, AwtWindow> earlyMap = earlyList.stream().collect(Collectors.toMap(AwtWindow::getId, w -> w));
        final Map<Integer, AwtWindow> laterMap = laterList.stream().collect(Collectors.toMap(AwtWindow::getId, w -> w));

        // added / changed / unchanged
        for (final AwtWindow laterWindow : laterList) {
          final Integer windowId = laterWindow.getId();
          final AwtWindow earlyWindow = earlyMap.get(windowId);

          if (earlyWindow == null) {
            // Window exists now, but didn't before -> added
            this.added.add(laterWindow);

          } else if (Objects.equals(earlyWindow.getType(), laterWindow.getType())) {
            // Only compare windows of the same id/type
            final AbstractAwtWindow.Difference difference = laterWindow.compare(earlyWindow, history);
            if (difference != null) {
              // Window exists now and before, but details differ -> changed
              this.changed.add(
                  new WindowsDifference.AwtWindowAndDifference(laterWindow, laterWindow.compare(earlyWindow, history)));

            } else {
              // Window exists now and before, and details are the same -> unchanged
              this.unchanged.add(laterWindow);
            }

          } else {
            // For two windows of different types, we don't compare them
            this.added.add(laterWindow);
            this.removed.add(earlyWindow);
          }
        }

        // removed
        for (final AwtWindow earlyWindow : earlyList) {
          final Integer windowId = earlyWindow.getId();
          if (!laterMap.containsKey(windowId)) {
            // Window existed before, but doesn't now -> removed
            this.removed.add(earlyWindow);
          }
        }
      }
    }

    public final LocalDateTime timestampEarly;
    public final LocalDateTime timestampLater;
    public final FromTo<String> currentImage;
    public final WindowsDifference windows;

    public Difference(final Snapshot early, final Snapshot later, final ImagejListener listener) {
      this.timestampEarly = early.timestamp;
      this.timestampLater = later.timestamp;
      this.currentImage = !Objects.equals(early.currentImage, later.currentImage)
          ? new FromTo<>(early.currentImage, later.currentImage)
          : null;

      final ImagejListener.HistoryResponse history = listener.getMessages(early.timestamp, later.timestamp);
      this.windows = new WindowsDifference(early.windows, later.windows, history);
    }
  }

  public static Difference compare(final Snapshot early, final Snapshot later, final ImagejListener listener) {
    return new Difference(early, later, listener);
  }

  public final int id;
  public final LocalDateTime timestamp;
  public final String currentImage; // Can be null if no window is active
  public final List<AwtWindow> windows;
  public final List<Action> actions;

  // The following fields usually not changed, so we don't capture them in the
  // snapshot. however, we add them as they are useful for LLM.
  public final int screenWidth;
  public final int screenHeight;
  public final String guiScale;

  private final WindowIdentifier identifier;

  public Snapshot(final LogService log, final WindowIdentifier identifier, final int id) {
    this.id = id;
    this.identifier = identifier;

    this.timestamp = LocalDateTime.now();

    final ImageWindow activeWindow = WindowManager.getCurrentWindow();
    this.currentImage = activeWindow != null ? activeWindow.getTitle() : null;

    this.windows = new ArrayList<>();
    for (final Window w : Window.getWindows()) {
      try {
        final AwtWindow window = AwtWindowProvider.create(this.identifier, w, log);
        if (window == null) {
          // This case should ideally not happen if AwtWindowProvider has a fallback (e.g.
          // UnknownWindow) Or, we might choose to log and skip if a truly unrepresentable
          // window is encountered.
          log.warn("Unsupported window type encountered and skipped: " + w.getClass().getName());
          continue;
        }
        windows.add(window);

      } catch (final AwtWindowProvider.InvalidWindowException e) {
        log.debug("Invalid window encountered and skipped: " + e.getMessage());
      }
    }
    this.actions = buildActions();

    final Dimension screen = IJ.getScreenSize();
    this.screenWidth = screen.width;
    this.screenHeight = screen.height;
    this.guiScale = IJ.d2s(Prefs.getGuiScale(), 2);
  }

  public void deactivate() {
    if (windows != null) {
      windows.forEach(AwtWindow::deactivate);
    }
  }

  public Action.Response runAction(final int actionId, final List<Object> parameters) {
    if (actionId < 0 || actionId >= actions.size()) {
      throw new IllegalArgumentException("Invalid action ID: " + actionId);
    }

    final Action action = actions.get(actionId);
    if (action == null) {
      throw new IllegalArgumentException("Action not found: " + actionId);
    }

    final List<String> path = new ArrayList<>(action.path);
    Collections.reverse(path);

    final String windowIdStr = path.remove(path.size() - 1); // remove window id from path
    final int targetWindowId = Integer.parseInt(windowIdStr);
    AwtWindow window = null;
    for (final AwtWindow w : this.windows) {
      if (w.getId() == targetWindowId) {
        window = w;
        break;
      }
    }
    if (window == null) {
      throw new IllegalArgumentException("Window not found with ID: " + targetWindowId);
    }

    final Object result = window.runAction(path, action.type, parameters);
    return new Action.Response(action.type, result);
  }

  private List<Action> buildActions() {
    final List<Action> actions = new ArrayList<>();
    for (final AwtWindow window : windows) {
      // window should not be null here as we populate the list carefully

      final List<Action> windowActions = window.getActions();
      if (windowActions != null && !windowActions.isEmpty()) {
        for (final Action action : windowActions) {
          action.path.add(String.valueOf(window.getId())); // Add window ID as string
          action.description = "Window #" + window.getId() + ": " + action.description; // Add window ID to description
          Collections.reverse(action.path); // Path is component -> window; reverse to make it window -> component
          actions.add(action);
        }
      }
    }
    return actions;
  }
}
