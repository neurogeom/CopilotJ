/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.awt.Component;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;

import ij.measure.ResultsTable;
import ij.text.TextWindow;

import copilotj.awt.Action;
import copilotj.awt.WindowIdentifier;
import copilotj.awt.window.IjTextWindow.ResultsTableChunk;

/** Uses a mock AWT window and real table data, so no display server is required. */
public class IjTextWindowTest {

  private static ResultsTable table(final int rows) {
    final ResultsTable table = new ResultsTable();
    for (int row = 0; row < rows; row++) {
      table.incrementCounter();
      table.addValue("Area", 256);
      table.addValue("X", 18 + row * 40);
    }
    return table;
  }

  private static IjTextWindow window(final ResultsTable table) {
    final TextWindow window = mock(TextWindow.class);
    when(window.getComponents()).thenReturn(new Component[0]);
    when(window.isShowing()).thenReturn(true);
    when(window.getResultsTable()).thenReturn(table);
    return new IjTextWindow(new WindowIdentifier(), window);
  }

  private static ResultsTableChunk read(final IjTextWindow window, final int offset, final int limit) {
    // Dispatch the exact type advertised to MCP clients, not a separately copied string.
    final Action published = window.getActions().get(0);
    return (ResultsTableChunk) window.runAction(Collections.emptyList(), published.type,
        Arrays.<Object>asList(offset, limit));
  }

  @Test
  public void publishedResultsTableActionReadsAllRows() {
    final ResultsTableChunk result = read(window(table(3)), 0, 10);
    assertEquals(3, result.total);
    assertEquals(0, result.offset);
    assertEquals(3, result.limit);
    assertArrayEquals(new Double[] { 256.0, 256.0, 256.0 }, result.data.get("Area"));
    assertArrayEquals(new Double[] { 18.0, 58.0, 98.0 }, result.data.get("X"));
  }

  @Test
  public void legacyResultsTableActionNameRemainsSupported() {
    final ResultsTableChunk result = (ResultsTableChunk) window(table(3)).runAction(
        Collections.emptyList(), "ij.text.TextWindow.get_results_table", Arrays.<Object>asList(1, 1));
    assertArrayEquals(new Double[] { 58.0 }, result.data.get("X"));
  }

  @Test
  public void nonzeroOffsetReturnsTheRequestedRows() {
    final ResultsTableChunk result = read(window(table(4)), 1, 2);
    assertEquals(1, result.offset);
    assertEquals(2, result.limit);
    assertArrayEquals(new Double[] { 58.0, 98.0 }, result.data.get("X"));
  }

  @Test
  public void finalPageStopsAtTheLastRow() {
    final ResultsTableChunk result = read(window(table(3)), 2, 10);
    assertEquals(1, result.limit);
    assertArrayEquals(new Double[] { 98.0 }, result.data.get("X"));
  }

  @Test
  public void offsetPastTheEndReturnsAnEmptyPage() {
    final ResultsTableChunk result = read(window(table(3)), 10, 10);
    assertEquals(3, result.total);
    assertEquals(3, result.offset);
    assertEquals(0, result.limit);
    assertArrayEquals(new Double[0], result.data.get("X"));
  }

  @Test
  public void defaultLimitIsBoundedByTheRemainingRows() {
    final ResultsTableChunk result = read(window(table(3)), 1, 0);
    assertEquals(2, result.limit);
    assertArrayEquals(new Double[] { 58.0, 98.0 }, result.data.get("X"));
  }

  @Test
  public void defaultLimitRemainsFiftyForLargerTables() {
    final ResultsTableChunk result = read(window(table(60)), 1, -1);
    assertEquals(50, result.limit);
    assertEquals(58.0, result.data.get("X")[0]);
    assertEquals(2018.0, result.data.get("X")[49]);
  }

  @Test
  public void negativeOffsetStartsAtTheFirstRow() {
    final ResultsTableChunk result = read(window(table(3)), -10, 1);
    assertEquals(0, result.offset);
    assertArrayEquals(new Double[] { 18.0 }, result.data.get("X"));
  }

  @Test
  public void largeLimitDoesNotOverflowThePageBoundary() {
    final ResultsTableChunk result = read(window(table(3)), 1, Integer.MAX_VALUE);
    assertEquals(2, result.limit);
    assertArrayEquals(new Double[] { 58.0, 98.0 }, result.data.get("X"));
  }

  @Test
  public void emptyTableHasNoRowsToAllocate() {
    final ResultsTableChunk result = read(window(table(0)), 0, 0);
    assertEquals(0, result.total);
    assertEquals(0, result.limit);
  }

  @Test
  public void missingResultsTableIsReported() {
    assertThrows(IllegalStateException.class, () -> read(window(null), 0, 10));
  }

  @Test
  public void resultsActionRejectsMalformedParameters() {
    final IjTextWindow window = window(table(3));
    final String type = window.getActions().get(0).type;
    final List<List<Object>> invalid = Arrays.asList(
        Collections.emptyList(), Collections.<Object>singletonList(0),
        Arrays.<Object>asList("0", 10), Arrays.<Object>asList(0, 1.5));
    for (final List<Object> parameters : invalid) {
      assertThrows(IllegalArgumentException.class,
          () -> window.runAction(Collections.emptyList(), type, parameters));
    }
  }
}
