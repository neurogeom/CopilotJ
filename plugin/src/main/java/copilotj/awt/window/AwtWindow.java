/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.util.List;

import copilotj.ImagejListener;
import copilotj.awt.Action;

public interface AwtWindow {
  public static class Difference {
    public final String type;
    public final int id;

    public Difference(final AwtWindow from, final AwtWindow to) {
      if (from == null || to == null) {
        throw new IllegalArgumentException("from and to must not be null");
      } else if (from.getId() != to.getId()) {
        throw new IllegalArgumentException("from and to must have the same id");
      } else if (!from.getType().equals(to.getType())) {
        throw new IllegalArgumentException("from and to must have the same type");
      }

      this.type = to.getType();
      this.id = to.getId();
    }
  }

  public int getId();

  public String getType();

  public Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history);

  public List<Action> getActions();

  public void deactivate();

  public Object runAction(final List<String> path, final String type, final List<Object> parameters);
}
