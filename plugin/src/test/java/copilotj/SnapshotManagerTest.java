/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;

import org.junit.jupiter.api.Test;
import org.scijava.log.LogService;

import copilotj.awt.Action;
import copilotj.awt.Snapshot;

/**
 * Tests {@link SnapshotManager}'s queue lifecycle: id assignment, FIFO eviction,
 * get/drop, and not-found errors. Uses the package-private injectable constructor (a
 * mock ImagejListener bypasses the real ctor's ij.* static registration) and overrides
 * {@code newSnapshot()} to return a mock Snapshot (the real Snapshot ctor hits
 * WindowManager/IJ statics). {@code get()}/{@code drop()} key on the raw {@code id}
 * field, so the mock's {@code id} is stamped via reflection.
 */
public class SnapshotManagerTest {

  /** A SnapshotManager whose newSnapshot() returns an id-stamped mock Snapshot. */
  static class TestSnapshotManager extends SnapshotManager {
    TestSnapshotManager() {
      super(mock(LogService.class), mock(ImagejListener.class), null);
    }

    @Override
    protected Snapshot newSnapshot(final int id) {
      final Snapshot snap = mock(Snapshot.class);
      setId(snap, id);
      return snap;
    }
  }

  private static void setId(final Snapshot snap, final int id) {
    try {
      final Field f = Snapshot.class.getField("id"); // public final int id
      f.setAccessible(true);
      f.setInt(snap, id);
    } catch (final ReflectiveOperationException e) {
      throw new AssertionError("unable to stamp Snapshot.id", e);
    }
  }

  /** A TestSnapshotManager whose created snapshots stub runAction, for happy-path tests. */
  static final class StubbingManager extends TestSnapshotManager {
    @Override
    protected Snapshot newSnapshot(final int id) {
      final Snapshot snap = super.newSnapshot(id);
      when(snap.runAction(anyInt(), any())).thenReturn(new Action.Response("ok", null));
      return snap;
    }
  }

  @Test
  public void captureAssignsIncrementingIds() {
    final SnapshotManager mgr = new TestSnapshotManager();
    assertEquals(1, mgr.capture().id);
    assertEquals(2, mgr.capture().id);
    assertEquals(3, mgr.capture().id);
  }

  @Test
  public void getReturnsSnapshotByIdOrNull() {
    final SnapshotManager mgr = new TestSnapshotManager();
    mgr.capture(); // id 1
    mgr.capture(); // id 2
    assertEquals(1, mgr.get(1).id);
    assertEquals(2, mgr.get(2).id);
    assertNull(mgr.get(99));
  }

  @Test
  public void dropRemovesSnapshot() {
    final SnapshotManager mgr = new TestSnapshotManager();
    mgr.capture(); // id 1
    mgr.capture(); // id 2
    mgr.drop(1);
    assertNull(mgr.get(1));
    assertNotNull(mgr.get(2));
  }

  @Test
  public void fifoEvictionAtMaxSnapshots() {
    final SnapshotManager mgr = new TestSnapshotManager();
    for (int i = 0; i < 17; i++) {
      mgr.capture();
    }
    assertNull(mgr.get(1), "oldest evicted once the 17th snapshot is stored");
    assertNotNull(mgr.get(17));
  }

  @Test
  public void compareThrowsWhenSnapshotNotFound() {
    final SnapshotManager mgr = new TestSnapshotManager();
    mgr.capture(); // id 1
    final SnapshotManager.CompareRequest req = new SnapshotManager.CompareRequest();
    req.idEarly = 99; // absent
    assertThrows(IllegalArgumentException.class, () -> mgr.compare(req));
  }

  @Test
  public void runActionThrowsWhenSnapshotNotFound() {
    final SnapshotManager mgr = new TestSnapshotManager();
    mgr.capture();
    final SnapshotManager.ActionRequest req = new SnapshotManager.ActionRequest();
    req.snapshotId = 99; // absent
    req.parameters = new ArrayList<>();
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(req));
  }

  @Test
  public void runActionReturnsCreatedSnapshotResult() {
    // Happy path: an existing snapshot is found, a fresh one is built via newSnapshot(),
    // and the action is delegated to it. Verifies the factory seam routes through runAction.
    final SnapshotManager mgr = new StubbingManager();
    mgr.capture(); // id 1, exists
    final SnapshotManager.ActionRequest req = new SnapshotManager.ActionRequest();
    req.snapshotId = 1;
    req.actionId = 0;
    req.parameters = new ArrayList<>();
    final Object result = mgr.runAction(req);
    assertNotNull(result);
    assertEquals("ok", ((Action.Response) result).type);
    // capture() used id 1, runAction() consumed id 2 for the created snapshot -> next is 3.
    assertEquals(3, mgr.capture().id);
  }
}
