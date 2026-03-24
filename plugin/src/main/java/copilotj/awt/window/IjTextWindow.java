/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import ij.measure.ResultsTable;
import ij.text.TextWindow;

import copilotj.ImagejListener;
import copilotj.awt.Action;
import copilotj.awt.WindowIdentifier;
import copilotj.util.FromTo;

public class IjTextWindow extends AbstractAwtWindow<TextWindow> {
  public static class Provider implements AwtWindowProvider {
    @Override
    public AwtWindow tryCreate(final WindowIdentifier identifier, final Window window) {
      if (window instanceof TextWindow) {
        return new IjTextWindow(identifier, (TextWindow) window);
      }
      return null;
    }
  }

  public static class ResultsTableSummary {
    public final String title;
    public final String[] headings;
    public final int size;

    public ResultsTableSummary(final ResultsTable t) {
      this.title = t.getTitle();
      this.headings = t.getHeadings();
      this.size = t.size();
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) {
        return true;
      } else if (o == null || getClass() != o.getClass()) {
        return false;
      }

      final ResultsTableSummary that = (ResultsTableSummary) o;
      return size == that.size && Objects.equals(title, that.title) && Arrays.equals(headings, that.headings);
    }

    public static class Difference {
      public final FromTo<String> title;
      public final FromTo<String[]> headings;
      public final FromTo<Integer> size;

      public Difference(final ResultsTableSummary from, final ResultsTableSummary to) {
        this.title = !Objects.equals(from.title, to.title) ? new FromTo<>(from.title, to.title) : null;
        this.headings = !Arrays.equals(from.headings, to.headings) ? new FromTo<>(from.headings, to.headings) : null;
        this.size = !Objects.equals(from.size, to.size) ? new FromTo<>(from.size, to.size) : null;
      }
    }
  }

  public static class ResultsTableChunk {
    public final String title;
    public final int total;
    public final int offset;
    public final int limit;
    public final Map<String, Double[]> data;

    public ResultsTableChunk(final ResultsTable t, int offset, int limit) {
      this.title = t.getTitle();
      this.total = t.size();

      if (offset < 0) {
        offset = 0;
      }

      if (limit <= 0) {
        limit = 50;
      } else if (offset + limit > t.size()) {
        limit = t.size() - offset;
      }

      this.offset = offset;
      this.limit = limit;
      this.data = new HashMap<>();

      final String[] headings = t.getHeadings();
      for (int i = 0; i < headings.length; i++) {
        final Double[] col = new Double[limit];
        for (int j = offset; j < limit; j++) {
          col[j] = t.getValue(headings[i], j);
        }
        this.data.put(headings[i], col);
      }
    }
  }

  public static class Difference extends AbstractAwtWindow.Difference {
    public final FromTo<String> title;
    public final ResultsTableSummary resultsTableAdded;
    public final ResultsTableSummary.Difference resultsTableChanged;
    public final boolean resultsTableRemoved;

    public Difference(final IjTextWindow from, final IjTextWindow to) {
      super(from, to);
      this.title = !Objects.equals(from.title, to.title) ? new FromTo<>(from.title, to.title) : null;
      if (from.resultsTable == null && to.resultsTable != null) {
        this.resultsTableAdded = to.resultsTable;
        this.resultsTableChanged = null;
        this.resultsTableRemoved = false;

      } else if (from.resultsTable != null && to.resultsTable == null) {
        this.resultsTableAdded = null;
        this.resultsTableChanged = null;
        this.resultsTableRemoved = true;

      } else if (from.resultsTable != null && to.resultsTable != null) {
        this.resultsTableAdded = null;
        this.resultsTableChanged = new ResultsTableSummary.Difference(from.resultsTable, to.resultsTable);
        this.resultsTableRemoved = false;

      } else {
        this.resultsTableAdded = null;
        this.resultsTableChanged = null;
        this.resultsTableRemoved = false;
      }
    }
  }

  private static final String TYPE = "ij.text.TextWindow";

  public final String title;
  public final ResultsTableSummary resultsTable;

  public IjTextWindow(final WindowIdentifier identifier, final TextWindow w) {
    super(identifier, w, TYPE);

    this.title = w.getTitle();

    final ResultsTable t = w.getResultsTable(); // Can be null
    this.resultsTable = t != null ? new ResultsTableSummary(t) : null;
  }

  @Override
  public boolean equals(final AwtWindow other) {
    if (!super.equals(other)) {
      return false;
    }

    final IjTextWindow that = (IjTextWindow) other;
    return Objects.equals(resultsTable, that.resultsTable) && Objects.equals(title, that.title);
  }

  @Override
  public AwtWindow.Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history) {
    if (!(from instanceof IjTextWindow)) {
      return super.compare(from, history);
    }

    final IjTextWindow fromTextWindow = (IjTextWindow) from;
    if (this.equals(fromTextWindow)) {
      return null; // No change
    }

    return new Difference(fromTextWindow, this);
  }

  public List<Action> getActions() {
    final Action getResultsTable = Action
        .builder(TYPE + ".getResultsTable", "Get Results Table", "Get detail rows of the results table")
        .addIntegerParameter("offset", "Offset of the rows") // TODO: default to 0
        .addIntegerParameter("limit", "Limit of the rows")
        .build();
    return Collections.singletonList(getResultsTable);
  }

  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Window is not active");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for IjImage");
    }

    switch (type) {
      case TYPE + ".get_results_table":
        if (parameters == null || parameters.size() != 2) {
          throw new IllegalArgumentException(
              "Action 'Get Results Table' requires exactly 2 parameters (offset, limit). Found: "
                  + (parameters != null ? parameters.size() : "null"));
        }

        final Object offset = parameters.get(0);
        if (!(offset instanceof Integer)) {
          throw new IllegalArgumentException(
              "Action 'Get Results Table' requires an int 'offset' parameter, but got " +
                  (offset != null ? offset.getClass().getSimpleName() : "null"));
        }

        final Object limit = parameters.get(1);
        if (!(limit instanceof Integer)) {
          throw new IllegalArgumentException(
              "Action 'Get Results Table' requires an int 'limit' parameter, but got " +
                  (limit != null ? limit.getClass().getSimpleName() : "null"));
        }

        return getResultsTable((Integer) offset, (Integer) limit);

      default:
        return super.runAction(path, type, parameters);
    }
  }

  public ResultsTableChunk getResultsTable(final int offset, final int limit) {
    final ResultsTable t = this.component.getResultsTable();
    if (t == null) {
      throw new IllegalStateException("No results table available");
    }

    return new ResultsTableChunk(t, offset, limit);
  }
}
