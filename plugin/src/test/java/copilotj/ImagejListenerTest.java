/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDateTime;
import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Tests {@link ImagejListener}'s pure message-queue logic: bounded FIFO eviction,
 * consecutive-message merging, and {@code getMessages(since, until)} filtering.
 *
 * <p>Uses the package-private {@code autoStart=false} constructor so no ImageJ listeners
 * are registered (the normal constructor calls {@code run(null)}, which hits ij.* statics).
 * The {@code ij.*}-calling listener callbacks (eventOccurred/imageOpened) are Layer C.
 */
public class ImagejListenerTest {

  private static ImagejListener newListener(final int maxSize) {
    return new ImagejListener(maxSize, false, false);
  }

  private static List<ImagejListener.Message> queue(final ImagejListener listener) {
    return listener.getMessages(LocalDateTime.MIN).messages;
  }

  // ===== bounded FIFO eviction =====

  @Test
  public void boundedQueueEvictsOldest() {
    final ImagejListener l = newListener(4);
    for (int i = 0; i < 5; i++) {
      l.push(new ImagejListener.LogMessage("m" + i));
    }
    final List<ImagejListener.Message> msgs = queue(l);
    assertEquals(4, msgs.size(), "capped at maxSize");
    assertEquals("m1", msgs.get(0).getMessage(), "oldest (m0) evicted FIFO");
    assertEquals("m4", msgs.get(3).getMessage());
  }

  // ===== consecutive-message merging =====

  @Test
  public void consecutiveEqualMessagesMerge() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("x"));
    l.push(new ImagejListener.LogMessage("x")); // merges into the previous
    l.push(new ImagejListener.LogMessage("y")); // distinct -> new entry

    final List<ImagejListener.Message> msgs = queue(l);
    assertEquals(2, msgs.size());
    assertEquals("x", msgs.get(0).getMessage());
    assertEquals(2, msgs.get(0).count, "merged message counts both pushes");
    assertNotNull(msgs.get(0).timestampLatest, "merge stamps timestampLatest");
    assertEquals("y", msgs.get(1).getMessage());
    assertEquals(1, msgs.get(1).count);
  }

  // ===== getMessages(since, until) =====

  @Test
  public void getMessagesThrowsOnNullSince() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("a"));
    assertThrows(IllegalArgumentException.class, () -> l.getMessages((LocalDateTime) null));
  }

  @Test
  public void getMessagesAtMinReturnsAllAndIsComplete() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("a"));
    l.push(new ImagejListener.LogMessage("b"));
    final ImagejListener.HistoryResponse r = l.getMessages(LocalDateTime.MIN);
    assertEquals(2, r.messages.size());
    assertTrue(r.isComplete, "oldest timestamp is after MIN -> history complete within window");
  }

  @Test
  public void getMessagesAtMaxReturnsNoneAndIsIncomplete() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("a"));
    final ImagejListener.HistoryResponse r = l.getMessages(LocalDateTime.MAX);
    assertEquals(0, r.messages.size());
    assertFalse(r.isComplete, "no message is after MAX -> not complete");
  }

  @Test
  public void getMessagesUntilFiltersByUpperBound() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("a"));
    l.push(new ImagejListener.LogMessage("b"));
    // until=MIN excludes everything (no timestamp is strictly before MIN)
    assertEquals(0, l.getMessages(LocalDateTime.MIN, LocalDateTime.MIN).messages.size());
    // until=MAX keeps everything
    assertEquals(2, l.getMessages(LocalDateTime.MIN, LocalDateTime.MAX).messages.size());
  }

  @Test
  public void getMessagesSinceInclusiveUntilStrict() {
    // Pin the boundary semantics directly: `since` is inclusive (>=), `until` is strict (<).
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("a"));
    final LocalDateTime ts = queue(l).get(0).timestampEarliest; // the message's own timestamp
    assertEquals(1, l.getMessages(ts).messages.size(), "since is inclusive: equal timestamp is kept");
    assertEquals(0, l.getMessages(LocalDateTime.MIN, ts).messages.size(),
        "until is strict: equal timestamp is excluded");
  }
}
