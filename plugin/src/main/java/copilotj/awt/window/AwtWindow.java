/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import copilotj.ImagejListener;

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

  public void deactivate();
}
