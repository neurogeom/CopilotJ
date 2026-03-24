/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Window;
import java.util.Objects;

import copilotj.ImagejListener;
import copilotj.awt.WindowIdentifier;
import copilotj.awt.container.AbstractContainerNode;

/**
 * Generic node for container types like Panel, Window, Frame, Dialog etc.
 *
 * @see https://docs.oracle.com/javase/8/docs/api/java/awt/Window.html
 */
public abstract class AbstractAwtWindow<T extends Window> extends AbstractContainerNode<T> implements AwtWindow {
  protected final int id;
  protected final WindowIdentifier identifier;

  public int getId() {
    return id;
  }

  public String getType() {
    return type;
  };

  public AbstractAwtWindow(final WindowIdentifier identifier, final T w, final String type) {
    super(type, w);
    this.id = identifier.getId(w);
    this.identifier = identifier;
  }

  public boolean equals(final AwtWindow other) {
    if (this == other) {
      return true;
    } else if (other == null || getClass() != other.getClass()) {
      return false;
    }

    return Objects.equals(this.id, other.getId()) && Objects.equals(this.type, other.getType());
  }

  public AwtWindow.Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history) {
    if (this.equals(from)) {
      return null;
    }

    return new AwtWindow.Difference(from, this);
  }
}
