/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;

import org.scijava.log.LogService;

import copilotj.awt.Action;
import copilotj.awt.ComponentIdentifier;
import copilotj.awt.Snapshot;
import copilotj.awt.WindowIdentifier;

public class SnapshotManager {
  public static class CompareRequest {
    public Integer idEarly;
    public Integer idLater = null;
  }

  public static class ActionRequest {
    public String ref;
    public String action;
    public List<Object> parameters;
  }

  private static final int MAX_SNAPSHOTS = 16;
  private final LogService log;
  private final ImagejListener listener;
  private final WindowIdentifier identifier;
  private final ComponentIdentifier componentIdentifier;

  private final Queue<Snapshot> snapshots = new ArrayDeque<>(MAX_SNAPSHOTS);
  private int nextId = 1;

  public SnapshotManager(final LogService log, final boolean debug) {
    this.log = log;
    this.listener = new ImagejListener(10240, debug);
    this.identifier = new WindowIdentifier();
    this.componentIdentifier = new ComponentIdentifier();
  }

  // FIXME: remove this method
  public ImagejListener getListener() {
    return listener;
  }

  // FIXME: remove this method
  public WindowIdentifier getIdentifier() {
    return identifier;
  }

  public Snapshot capture() {
    return capture(true);
  }

  private Snapshot capture(final boolean save) {
    final int id = nextId++;
    final Snapshot snapshot = new Snapshot(log, identifier, componentIdentifier, id);
    if (!save) {
      return snapshot;
    }

    if (snapshots.size() >= MAX_SNAPSHOTS) {
      snapshots.poll(); // Remove oldest snapshot
    }
    snapshot.deactivate(); // deactivate to minimize memory usage
    snapshots.add(snapshot);
    return snapshot;
  }

  public Snapshot get(final int id) {
    // This can be optimised by using binary search
    for (final Snapshot snapshot : snapshots) {
      if (snapshot.id == id) {
        return snapshot;
      }
    }
    return null;
  }

  public void drop(final int id) {
    snapshots.removeIf(snapshot -> snapshot.id == id);
  }

  public Snapshot.Difference compare(final CompareRequest request) {
    final Snapshot early = get(request.idEarly);
    final Snapshot later = request.idLater != null ? get(request.idLater) : capture(false);
    if (early == null || later == null) {
      throw new IllegalArgumentException("Snapshot not found");
    }

    return Snapshot.compare(early, later, listener);
  }

  public Action.Response runAction(final ActionRequest request) {
    // Actions always run against a freshly captured snapshot: saved snapshots are
    // deactivated (their live AWT components are nulled), and refs are stable
    // identities, so the same ref resolves to the same component on the fresh
    // snapshot. A missing ref yields a clean "not found" error.
    final Snapshot fresh = new Snapshot(log, identifier, componentIdentifier, nextId++);
    return fresh.runAction(request.ref, request.action, request.parameters);
  }
}
