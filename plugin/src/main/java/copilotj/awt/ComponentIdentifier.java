/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt;

import java.awt.Component;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Assigns stable, playwright-mcp-style refs ({@code "e" + int}) to AWT
 * components, keyed on the live {@link Component} instance.
 *
 * <p>
 * The mapping is held in a {@link WeakHashMap}, so it does not prevent components
 * from being garbage collected, and it persists across snapshots as long as the
 * component stays alive on screen. The same component always receives the same
 * ref; a brand-new component receives the next monotonically increasing ref; a
 * component that was closed/GC'd has its ref vanish (callers then get a clean
 * "ref not found" error instead of silently targeting a different element).
 * </p>
 *
 * <p>
 * This is the component-level sibling of {@link WindowIdentifier}. Both live
 * once in {@code SnapshotManager} so their state survives across snapshots.
 * </p>
 */
public class ComponentIdentifier {

  private final Map<Component, Integer> componentToRef = new WeakHashMap<>();
  private final AtomicInteger nextRef = new AtomicInteger(0);

  /**
   * Gets the stable ref number for the given AWT component, allocating a new one
   * on first sight.
   *
   * @param component The AWT component. Must not be null.
   * @return A unique integer ref for the component.
   * @throws IllegalArgumentException if the component is null.
   */
  public synchronized int getRef(final Component component) {
    if (component == null) {
      throw new IllegalArgumentException("Component cannot be null");
    }
    // computeIfAbsent ensures the ref is generated only once per component and
    // that the operation is atomic under the synchronized method. The
    // WeakHashMap lets the entry be reclaimed once the component is GC'd.
    return componentToRef.computeIfAbsent(component, k -> nextRef.getAndIncrement());
  }
}
