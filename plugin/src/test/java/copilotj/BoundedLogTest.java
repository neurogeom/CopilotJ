/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link BoundedLog}, a synchronized append-only log with a soft
 * char cap. Lives in package {@code copilotj} so it can reference the
 * package-private {@link BoundedLog#MAX_CHARS} directly.
 */
public class BoundedLogTest {

  @Test
  public void snapshotEmptyAfterConstruction() {
    assertEquals("", new BoundedLog().snapshot());
  }

  @Test
  public void appendIgnoresNull() {
    final BoundedLog log = new BoundedLog();
    log.append(null);
    assertEquals("", log.snapshot());
  }

  @Test
  public void appendIgnoresEmpty() {
    final BoundedLog log = new BoundedLog();
    log.append("");
    assertEquals("", log.snapshot());
  }

  @Test
  public void appendAndSnapshotJoinsInOrder() {
    final BoundedLog log = new BoundedLog();
    log.append("a");
    log.append("b");
    log.append("c");
    assertEquals("abc", log.snapshot());
  }

  @Test
  public void clearEmptiesLog() {
    final BoundedLog log = new BoundedLog();
    log.append("x");
    log.clear();
    assertEquals("", log.snapshot());
  }

  @Test
  public void evictsOldestChunksOverCap() {
    // Four MAX_CHARS/4 chunks fill exactly to the cap; a fifth evicts the oldest.
    final BoundedLog log = new BoundedLog();
    final int chunk = BoundedLog.MAX_CHARS / 4;
    log.append(repeat('1', chunk));
    log.append(repeat('2', chunk));
    log.append(repeat('3', chunk));
    log.append(repeat('4', chunk));
    final String newest = repeat('5', chunk);
    log.append(newest);

    final String snap = log.snapshot();
    assertTrue(snap.endsWith(newest), "newest chunk must be fully retained");
    assertFalse(snap.startsWith(repeat('1', chunk)), "oldest chunk must be evicted");
    assertTrue(snap.length() <= BoundedLog.MAX_CHARS, "total retained must stay under the cap");
  }

  @Test
  public void alwaysKeepsAtLeastOneChunkEvenWhenOverCap() {
    // A single append larger than MAX_CHARS is never evicted (size > 1 guard).
    final BoundedLog log = new BoundedLog();
    final String huge = repeat('z', BoundedLog.MAX_CHARS + 100);
    log.append(huge);
    assertEquals(huge.length(), log.snapshot().length());
  }

  private static String repeat(final char c, final int n) {
    final char[] arr = new char[n];
    Arrays.fill(arr, c);
    return new String(arr);
  }
}
