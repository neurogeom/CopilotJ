/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import java.awt.EventQueue;
import java.time.LocalDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import ij.CommandListener;
import ij.Executer;
import ij.IJ;
import ij.IJEventListener;
import ij.ImageListener;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.gui.RoiListener;
import ij.gui.Toolbar;
import ij.plugin.PlugIn;

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import copilotj.util.FlexibleLocalDateTimeDeserializer;

/**
 * This plugin implements the Plugins/Utilities/Monitor Events command.
 * By implementing the IJEventListener, CommandListener, ImageListener
 * and RoiListener interfaces, it is able to monitor foreground and background
 * color changes, tool switches, Log window closings, command executions, image
 * window openings, closings and updates, and ROI changes.
 */
public class ImagejListener implements PlugIn, IJEventListener, ImageListener, RoiListener, CommandListener {
  // --- Messages ---

  public abstract static class Message {
    public final LocalDateTime timestampEarliest;
    public LocalDateTime timestampLatest;
    public int count;

    public Message() {
      this.timestampEarliest = LocalDateTime.now();
      this.timestampLatest = null;
      this.count = 1;
    }

    public boolean mergeIfEquals(final Message other) {
      if (this.getClass() == other.getClass() && Objects.equals(this.getMessage(), other.getMessage())) {
        this.timestampLatest = LocalDateTime.now();
        this.count += 1;
        return true;
      }
      return false;
    }

    public abstract String getMessage();
  }

  public static class LogMessage extends Message {
    private final String text;

    public LogMessage(final String text) {
      super();
      this.text = text;
    }

    @Override
    public String getMessage() {
      return this.text;
    }
  }

  public static class HistoryResponse {
    public final List<Message> messages;
    public final boolean isComplete;
    public final LocalDateTime since;
    public final LocalDateTime until;

    private Set<String> updatedTitles;

    public HistoryResponse(final List<Message> messages, final boolean isComplete, final LocalDateTime since,
        final LocalDateTime until) {
      this.messages = messages;
      this.isComplete = isComplete;
      this.since = since;
      this.until = until;
    }

    public boolean isImageUpdated(final String title) {
      if (updatedTitles == null) {
        updateUpdatedImages();
      }
      return updatedTitles.contains(title);
    }

    /*
     * Collect titles of updated images from messages
     */
    private void updateUpdatedImages() {
      updatedTitles = new HashSet<>();

      for (final ImagejListener.Message message : messages) {
        if (message instanceof ImagejListener.ImageMessage) {
          final ImagejListener.ImageMessage imgMsg = (ImagejListener.ImageMessage) message;
          // Note: imgMsg.edt likely refers to the Event Dispatch Thread,
          // but is not currently used in this comparison logic.
          if (imgMsg.type == ImagejListener.ImageMessage.Type.UPDATED) {
            updatedTitles.add(imgMsg.title);
          }
        }
      }
    }
  }

  public static class HistoryRequest {
    @JsonDeserialize(using = FlexibleLocalDateTimeDeserializer.class)
    public LocalDateTime since;
  }

  public static class ColorChangeMessage extends Message {
    private final String type;
    private final java.awt.Color color;

    public ColorChangeMessage(final String type, final java.awt.Color color) {
      super();
      this.type = type;
      this.color = color;
    }

    @Override
    public String getMessage() {
      String hex = Integer.toHexString(color.getRGB());
      hex = "#" + hex.substring(2);
      return "Changed " + type + " color to " + hex;
    }
  }

  public static class ToolChangeMessage extends Message {
    private final String toolName;

    public ToolChangeMessage(final String toolName) {
      super();
      this.toolName = toolName;
    }

    @Override
    public String getMessage() {
      return "Switched to the " + toolName + (toolName.endsWith("Tool") ? "" : " tool");
    }
  }

  public static class ColorPickerClosedMessage extends Message {
    @Override
    public String getMessage() {
      return "Color picker closed";
    }
  }

  public static class LogWindowClosedMessage extends Message {
    @Override
    public String getMessage() {
      return "Log window closed";
    }
  }

  public static class ImageMessage extends Message {
    public static enum Type {
      OPENED, CLOSED, UPDATED, SAVED
    }

    public final Type type;
    public final String title;
    public final boolean edt;

    public ImageMessage(final Type type, final String title, final boolean edt) {
      super();
      this.type = type;
      this.title = title;
      this.edt = edt;
    }

    @Override
    public String getMessage() {
      return type + " image: \"" + title + "\"" + (edt ? " (EDT)" : " (not EDT)");
    }
  }

  public static class RoiMessage extends Message {
    public static enum Type {
      CREATED, MOVED, MODIFIED, EXTENDED, COMPLETED, DELETED
    }

    public final Type type;
    public final String imageTitle;

    public RoiMessage(final Type type, final String imageTitle) {
      super();
      this.type = type;
      this.imageTitle = imageTitle;
    }

    @Override
    public String getMessage() {
      return "ROI modified: " + (imageTitle != null ? imageTitle : "") + ", " + type;
    }
  }

  public static class CommandExecutedMessage extends Message {
    private final String command;

    public CommandExecutedMessage(final String command) {
      super();
      this.command = command;
    }

    @Override
    public String getMessage() {
      return "Command executed: \"" + command + "\" command";
    }
  }

  // --- Base ---

  private final boolean debug;
  private final int maxSize;
  private final Deque<Message> messageQueue = new ArrayDeque<>();

  public ImagejListener() {
    this(10240, false);
  }

  public ImagejListener(final int maxSize, final boolean debug) {
    this(maxSize, debug, true);
  }

  // Package-private: lets unit tests construct a non-registering listener
  // (autoStart=false) so the pure message-queue logic can be exercised without
  // triggering ImageJ static registration (run(null) calls ij.* statics).
  ImagejListener(final int maxSize, final boolean debug, final boolean autoStart) {
    this.debug = debug;
    this.maxSize = maxSize;
    if (autoStart) {
      this.run(null);
    }
  }

  // --- Message Management ---

  void push(final String message) {
    push(new LogMessage(message));
  }

  void push(final Message message) {
    if (debug) {
      System.out.println(LocalDateTime.now() + ": " + message.getMessage());
    }

    if (!messageQueue.isEmpty()) {
      // peek last message and try to merge
      final Message lastMessage = messageQueue.peekLast();
      if (lastMessage.mergeIfEquals(message)) {
        return;
      }
    }

    messageQueue.add(message);

    if (messageQueue.size() > maxSize) {
      messageQueue.remove(); // Keep the queue size limited
    }
  }

  // --- Listener: IJEventListener ---

  @Override
  public void run(final String _args) {
    IJ.addEventListener(this);
    Executer.addCommandListener(this);
    ImagePlus.addImageListener(this);
    Roi.addRoiListener(this);
    push("EventListener started");
  }

  public void stop() {
    IJ.removeEventListener(this);
    Executer.removeCommandListener(this);
    ImagePlus.removeImageListener(this);
    Roi.removeRoiListener(this);
    push("EventListener stopped");
  }

  public HistoryResponse getMessages(final HistoryRequest request) {
    return getMessages(request.since);
  }

  public HistoryResponse getMessages(final LocalDateTime since) {
    return getMessages(since, null);
  }

  public HistoryResponse getMessages(final LocalDateTime since, final LocalDateTime until) {
    if (since == null) {
      throw new IllegalArgumentException("since is required");
    }

    // TODO: perf, use ring buffer
    final List<Message> messages = new ArrayList<>();
    boolean isComplete = false;

    // Check if we have messages older than requested time
    if (!messageQueue.isEmpty()) {
      final Message oldestMessage = messageQueue.peekFirst();
      if (oldestMessage.timestampEarliest.isAfter(since)) {
        isComplete = true;
      }
    }

    // Collect matching messages
    for (final Message message : messageQueue) {
      if ((message.timestampEarliest.isAfter(since) || message.timestampEarliest.isEqual(since)) &&
          (until == null || message.timestampEarliest.isBefore(until))) {
        messages.add(message);
      }
    }
    return new HistoryResponse(messages, isComplete, since, until);
  }

  // --- Listener: ImageListener ---

  @Override
  public void eventOccurred(final int eventID) {
    switch (eventID) {
      case IJEventListener.FOREGROUND_COLOR_CHANGED:
        push(new ColorChangeMessage("foreground", Toolbar.getForegroundColor()));
        break;

      case IJEventListener.BACKGROUND_COLOR_CHANGED:
        push(new ColorChangeMessage("background", Toolbar.getBackgroundColor()));
        break;

      case IJEventListener.TOOL_CHANGED:
        push(new ToolChangeMessage(IJ.getToolName()));
        break;

      case IJEventListener.COLOR_PICKER_CLOSED:
        push(new ColorPickerClosedMessage());
        break;

      case IJEventListener.LOG_WINDOW_CLOSED:
        push(new LogWindowClosedMessage());
        break;
    }
  }

  // called when an image is opened
  @Override
  public void imageOpened(final ImagePlus imp) {
    push(new ImageMessage(ImageMessage.Type.OPENED, imp.getTitle(), EventQueue.isDispatchThread()));
  }

  @Override
  public void imageClosed(final ImagePlus imp) {
    push(new ImageMessage(ImageMessage.Type.CLOSED, imp.getTitle(), EventQueue.isDispatchThread()));
  }

  @Override
  public void imageUpdated(final ImagePlus imp) {
    push(new ImageMessage(ImageMessage.Type.UPDATED, imp.getTitle(), EventQueue.isDispatchThread()));
  }

  public void imageSaved(final ImagePlus imp) {
    push(new ImageMessage(ImageMessage.Type.SAVED, imp.getTitle(), EventQueue.isDispatchThread()));
  }

  // --- Listener: RoiListener ---

  @Override
  public void roiModified(final ImagePlus img, final int id) {
    RoiMessage.Type type;
    switch (id) {
      case CREATED:
        type = RoiMessage.Type.CREATED;
        break;
      case MOVED:
        type = RoiMessage.Type.MOVED;
        break;
      case MODIFIED:
        type = RoiMessage.Type.MODIFIED;
        break;
      case EXTENDED:
        type = RoiMessage.Type.EXTENDED;
        break;
      case COMPLETED:
        type = RoiMessage.Type.COMPLETED;
        break;
      case DELETED:
        type = RoiMessage.Type.DELETED;
        break;
      default:
        throw new IllegalArgumentException("Unknown ROI modification type: " + id);
    }
    push(new RoiMessage(type, img != null ? img.getTitle() : ""));
  }

  // --- Listener: CommandListener ---

  @Override
  public String commandExecuting(final String command) {
    push(new CommandExecutedMessage(command));
    return command;
  }
}
