/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;

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

  @Test
  public void getMessagesReturnsSnapshotsUnaffectedByLaterMerges() {
    final ImagejListener l = newListener(16);
    l.push(new ImagejListener.LogMessage("x"));
    final ImagejListener.Message before = queue(l).get(0);

    l.push(new ImagejListener.LogMessage("x")); // merges into the queued message

    assertEquals(1, before.count, "a returned message must not see later merges");
    assertNull(before.timestampLatest, "a returned message must not see later merges");
    assertEquals("x", before.getMessage(), "snapshot keeps subclass state");
    assertEquals(2, queue(l).get(0).count, "the queued message itself is still merged");
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

  // ===== concurrent access =====
  // ImageJ calls the listener from macro/script threads and the EDT at the same time.

  private static final int THREADS = 8;
  private static final int PUSHES_PER_THREAD = 5000;

  /** Runs {@code task(threadIndex)} on THREADS threads released together; returns the first failure. */
  private static Throwable runConcurrently(final IntConsumer task) throws InterruptedException {
    final CountDownLatch start = new CountDownLatch(1);
    final AtomicReference<Throwable> failure = new AtomicReference<>();
    final Thread[] threads = new Thread[THREADS];
    for (int t = 0; t < THREADS; t++) {
      final int index = t;
      threads[t] = new Thread(() -> {
        try {
          start.await();
          task.accept(index);
        } catch (final Throwable e) {
          failure.compareAndSet(null, e);
        }
      });
      threads[t].start();
    }
    start.countDown();
    for (final Thread thread : threads) {
      thread.join();
    }
    return failure.get();
  }

  @Test
  public void concurrentPushesAreNotLost() throws InterruptedException {
    final ImagejListener l = newListener(THREADS * PUSHES_PER_THREAD);
    final Throwable failure = runConcurrently(t -> {
      for (int i = 0; i < PUSHES_PER_THREAD; i++) {
        l.push(new ImagejListener.LogMessage("t" + t + "-" + i)); // distinct: never merged
      }
    });
    assertNull(failure, "push must not throw under concurrent use");
    assertEquals(THREADS * PUSHES_PER_THREAD, queue(l).size(), "no push may be lost");
  }

  @Test
  public void concurrentMergesCountEveryPush() throws InterruptedException {
    final ImagejListener l = newListener(16);
    final Throwable failure = runConcurrently(t -> {
      for (int i = 0; i < PUSHES_PER_THREAD; i++) {
        l.push(new ImagejListener.LogMessage("same")); // always merged into the tail
      }
    });
    assertNull(failure, "push must not throw under concurrent use");
    final List<ImagejListener.Message> msgs = queue(l);
    assertEquals(1, msgs.size());
    assertEquals(THREADS * PUSHES_PER_THREAD, msgs.get(0).count, "every merge must be counted");
  }

  @Test
  public void getMessagesIsSafeDuringConcurrentPushes() throws InterruptedException {
    final ImagejListener l = newListener(64); // small cap: pushes evict while readers iterate
    final Throwable failure = runConcurrently(t -> {
      for (int i = 0; i < PUSHES_PER_THREAD; i++) {
        if (t % 2 == 0) {
          l.push(new ImagejListener.LogMessage("t" + t + "-" + i));
        } else {
          l.getMessages(LocalDateTime.MIN);
        }
      }
    });
    assertNull(failure, "getMessages must not throw while other threads push");
    assertEquals(64, queue(l).size());
  }
}
