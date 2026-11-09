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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.scijava.log.LogService;

import copilotj.awt.Action;
import copilotj.awt.Snapshot;
import copilotj.awt.window.AwtWindow;

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
    setField(snap, "id", id);
  }

  private static void setField(final Snapshot snap, final String name, final Object value) {
    try {
      final Field f = Snapshot.class.getField(name); // public final
      f.setAccessible(true);
      f.set(snap, value);
    } catch (final ReflectiveOperationException e) {
      throw new AssertionError("unable to set Snapshot." + name, e);
    }
  }

  /** A TestSnapshotManager whose created snapshots stub runAction, for happy-path tests. */
  static final class StubbingManager extends TestSnapshotManager {
    @Override
    protected Snapshot newSnapshot(final int id) {
      final Snapshot snap = super.newSnapshot(id);
      setField(snap, "actions", Collections.singletonList(action(10, "table.read")));
      when(snap.runAction(any(Action.class), any())).thenReturn(new Action.Response("ok", null));
      return snap;
    }
  }

  private static Action action(final int windowId, final String type) {
    final Action action = Action.builder(type, "Read", "Read window data").build();
    action.addPath(String.valueOf(windowId));
    return action;
  }

  /** Executes real Snapshot routing against mock windows, without creating an AWT UI. */
  static final class RoutingManager extends SnapshotManager {
    private final List<List<Action>> states;
    private int nextState;

    RoutingManager(final List<Action> captured, final List<Action> current) {
      super(mock(LogService.class), mock(ImagejListener.class), null);
      this.states = Arrays.asList(captured, current);
    }

    @Override
    protected Snapshot newSnapshot(final int id) {
      final Snapshot snapshot = mock(Snapshot.class, CALLS_REAL_METHODS);
      final List<Action> actions = states.get(nextState++);
      final List<AwtWindow> windows = new ArrayList<>();
      for (final Action action : actions) {
        final int windowId = Integer.parseInt(action.path.get(0));
        final AwtWindow window = mock(AwtWindow.class);
        when(window.getId()).thenReturn(windowId);
        when(window.runAction(anyList(), anyString(), any())).thenReturn("window-" + windowId);
        windows.add(window);
      }
      setId(snapshot, id);
      setField(snapshot, "actions", actions);
      setField(snapshot, "windows", windows);
      return snapshot;
    }
  }

  private static SnapshotManager.ActionRequest request(final int snapshotId, final int actionId) {
    final SnapshotManager.ActionRequest request = new SnapshotManager.ActionRequest();
    request.snapshotId = snapshotId;
    request.actionId = actionId;
    request.parameters = Arrays.<Object>asList(0, 10);
    return request;
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

  @Test
  public void selectedActionSurvivesEarlierWindowClosing() {
    // Old index 1 names window 20. After window 10 closes, index 1 names window 30.
    final SnapshotManager mgr = new RoutingManager(
        Arrays.asList(action(10, "table.read"), action(20, "table.read")),
        Arrays.asList(action(20, "table.read"), action(30, "table.read")));
    final int snapshotId = mgr.capture().id;
    final Action.Response result = mgr.runAction(request(snapshotId, 1));
    assertEquals("window-20", result.result);
  }

  @Test
  public void selectedActionSurvivesAnEarlierActionBeingInserted() {
    final SnapshotManager mgr = new RoutingManager(
        Collections.singletonList(action(20, "table.read")),
        Arrays.asList(action(10, "table.read"), action(20, "table.read")));
    final int snapshotId = mgr.capture().id;
    assertEquals("window-20", mgr.runAction(request(snapshotId, 0)).result);
  }

  @Test
  public void closedTargetIsRejectedInsteadOfReadingAnotherWindow() {
    final SnapshotManager mgr = new RoutingManager(
        Collections.singletonList(action(20, "table.read")),
        Collections.singletonList(action(30, "table.read")));
    final int snapshotId = mgr.capture().id;
    final IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
        () -> mgr.runAction(request(snapshotId, 0)));
    assertTrue(error.getMessage().contains("take a new snapshot"));
  }

  @Test
  public void changedActionTypeAtTheSamePathIsRejected() {
    final SnapshotManager mgr = new RoutingManager(
        Collections.singletonList(action(20, "table.read")),
        Collections.singletonList(action(20, "window.close")));
    final int snapshotId = mgr.capture().id;
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(request(snapshotId, 0)));
  }

  @Test
  public void changedComponentPathIsRejected() {
    final Action captured = action(20, "button.click");
    captured.addPath("children[0]");
    final Action current = action(20, "button.click");
    current.addPath("children[1]");
    final SnapshotManager mgr = new RoutingManager(Collections.singletonList(captured),
        Collections.singletonList(current));
    final int snapshotId = mgr.capture().id;
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(request(snapshotId, 0)));
  }

  @Test
  public void imageCaptureActionKeepsItsSelectedWindow() {
    final SnapshotManager mgr = new RoutingManager(
        Collections.singletonList(action(20, "ij.Image.capture")),
        Arrays.asList(action(10, "table.read"), action(20, "ij.Image.capture")));
    final int snapshotId = mgr.capture().id;
    final Action.Response result = mgr.runAction(request(snapshotId, 0));
    assertEquals("ij.Image.capture", result.type);
    assertEquals("window-20", result.result);
  }

  @Test
  public void actionIndexIsValidatedAgainstTheRequestedSnapshot() {
    final SnapshotManager mgr = new RoutingManager(
        Collections.singletonList(action(20, "table.read")),
        Arrays.asList(action(10, "table.read"), action(20, "table.read")));
    final int snapshotId = mgr.capture().id;
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(request(snapshotId, 1)));
  }

  @Test
  public void negativeActionIndexIsRejected() {
    final SnapshotManager mgr = new StubbingManager();
    final int snapshotId = mgr.capture().id;
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(request(snapshotId, -1)));
  }

  @Test
  public void actionFromAnEvictedSnapshotIsRejected() {
    final SnapshotManager mgr = new StubbingManager();
    final int expiredId = mgr.capture().id;
    for (int i = 0; i < 16; i++) {
      mgr.capture();
    }
    assertThrows(IllegalArgumentException.class, () -> mgr.runAction(request(expiredId, 0)));
  }
}
