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
import java.util.HashMap;
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
import copilotj.awt.component.ComponentNode;
import copilotj.awt.container.ContainerNode;
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

  // The following fields usually not changed, so we don't capture them in the
  // snapshot. however, we add them as they are useful for LLM.
  public final int screenWidth;
  public final int screenHeight;
  public final String guiScale;

  private final WindowIdentifier identifier;
  private final ComponentIdentifier componentIdentifier;

  // Ref handle -> node, populated while the AWT components are still live (in the
  // constructor, before deactivate()). Used to resolve call_action(ref, ...).
  private final Map<String, ComponentNode> refToNode = new HashMap<>();

  public Snapshot(final LogService log, final WindowIdentifier identifier,
      final ComponentIdentifier componentIdentifier, final int id) {
    this.id = id;
    this.identifier = identifier;
    this.componentIdentifier = componentIdentifier;

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
    assignRefs();

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

  /**
   * Walks the window/component tree depth-first and assigns stable ref handles
   * (via the session-scoped {@link ComponentIdentifier}) to every ref-eligible
   * node, populating {@link #refToNode}. Must run while the AWT components are
   * still live (i.e. in the constructor, before {@link #deactivate()}).
   */
  private void assignRefs() {
    for (final AwtWindow window : windows) {
      assignRefs((ComponentNode) window);
    }
  }

  private void assignRefs(final ComponentNode node) {
    node.assignRef(componentIdentifier);
    if (node.getRef() != null) {
      refToNode.put(node.getRef(), node);
    }
    if (node instanceof ContainerNode) {
      final ContainerNode container = (ContainerNode) node;
      final List<ComponentNode> children = container.getChildren();
      if (children != null) {
        for (final ComponentNode child : children) {
          assignRefs(child);
        }
      }
    }
  }

  /**
   * Resolves a ref handle to its node in this snapshot, or {@code null} if the
   * ref is not present (e.g. the element was closed since the snapshot the
   * caller holds).
   *
   * @param ref the ref handle (e.g. {@code "e5"}).
   * @return the node, or {@code null}.
   */
  public ComponentNode nodeByRef(final String ref) {
    return refToNode.get(ref);
  }

  /**
   * Runs an action on the node identified by {@code ref}, dispatching by the
   * short action id (e.g. {@code click}, {@code selectItem}).
   *
   * @param ref the ref handle of the target node.
   * @param action the short action id.
   * @param parameters the action parameters, or {@code null}.
   * @return the action response.
   * @throws IllegalArgumentException if the ref is not found in this snapshot.
   */
  public Action.Response runAction(final String ref, final String action, final List<Object> parameters) {
    final ComponentNode node = refToNode.get(ref);
    if (node == null) {
      throw new IllegalArgumentException(
          "Ref " + ref + " not found in the current snapshot. Try capturing a new snapshot.");
    }
    final Object result = node.runAction(action, parameters);
    return new Action.Response(resolveActionType(node, action), result);
  }

  /**
   * Resolves the short action id to its fully-qualified action type, falling back
   * to the short id if the node exposes no matching action. The response must
   * carry the fully-qualified type so clients (e.g. the Python bridge's
   * TypedActionResponse union) can discriminate on it.
   */
  private static String resolveActionType(final ComponentNode node, final String action) {
    final List<Action> actions = node.getActions();
    if (actions != null) {
      for (final Action a : actions) {
        if (a.shortId().equals(action)) {
          return a.type;
        }
      }
    }
    return action;
  }
}
