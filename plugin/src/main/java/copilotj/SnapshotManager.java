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
import copilotj.awt.Snapshot;
import copilotj.awt.WindowIdentifier;

public class SnapshotManager {
  public static class CompareRequest {
    public Integer idEarly;
    public Integer idLater = null;
  }

  public static class ActionRequest {
    public int snapshotId;
    public int actionId;
    public List<Object> parameters;
  }

  private static final int MAX_SNAPSHOTS = 16;
  private final LogService log;
  private final ImagejListener listener;
  private final WindowIdentifier identifier;

  private final Queue<Snapshot> snapshots = new ArrayDeque<>(MAX_SNAPSHOTS);
  private int nextId = 1;

  public SnapshotManager(final LogService log, final boolean debug) {
    this(log, new ImagejListener(10240, debug), new WindowIdentifier());
  }

  // Package-private: lets tests inject mock collaborators (a mock ImagejListener bypasses
  // the real constructor's ij.* static registration) and override newSnapshot().
  SnapshotManager(final LogService log, final ImagejListener listener,
      final WindowIdentifier identifier) {
    this.log = log;
    this.listener = listener;
    this.identifier = identifier;
  }

  // Overridable factory so tests can substitute a Snapshot double without invoking the
  // real Snapshot constructor (which hits WindowManager/IJ statics).
  protected Snapshot newSnapshot(final int id) {
    return new Snapshot(log, identifier, id);
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
    final Snapshot snapshot = newSnapshot(id);
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
    final Snapshot snapshot = this.get(request.snapshotId);
    if (snapshot == null) {
      throw new IllegalArgumentException("Snapshot not found");
    }

    // TODO: use old snapshot as reference
    final int id = nextId++;
    final Snapshot created = newSnapshot(id);
    return created.runAction(request.actionId, request.parameters);
  }
}
