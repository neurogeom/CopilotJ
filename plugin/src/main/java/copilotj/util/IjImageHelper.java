/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Base64;

import javax.imageio.ImageIO;

import ij.IJ;

public class IjImageHelper {
  /** Saves a captured Image as a PNG file. */
  public static void saveImage(final java.awt.Image img, final String filePath) {
    try {
      final BufferedImage bufferedImage = (BufferedImage) img;
      final File file = new File(filePath);
      ImageIO.write(bufferedImage, "png", file);
      IJ.log("Image saved: " + filePath);
    } catch (final Exception e) {
      IJ.log("Error saving image: " + e.getMessage());
    }
  }

  public static BufferedImage downsizeImage(final BufferedImage image, final int maxWidth, final int maxHeight) {
    final int originalWidth = image.getWidth();
    final int originalHeight = image.getHeight();

    // Check if resizing is needed
    if (originalWidth <= maxWidth && originalHeight <= maxHeight) {
      return image; // No need to resize
    }

    // Calculate the new size while keeping aspect ratio
    final double scaleFactor = Math.min((double) maxWidth / originalWidth, (double) maxHeight / originalHeight);
    final int newWidth = (int) (originalWidth * scaleFactor);
    final int newHeight = (int) (originalHeight * scaleFactor);

    // Resize the image
    final java.awt.Image resized = image.getScaledInstance(newWidth, newHeight, java.awt.Image.SCALE_SMOOTH);
    final BufferedImage newImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_ARGB);
    final Graphics2D g2d = newImage.createGraphics();
    g2d.drawImage(resized, 0, 0, null);
    g2d.dispose();

    return newImage;
  }

  public static String encodeImageToBase64(final BufferedImage image, final String format) {
    try {
      final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
      ImageIO.write(image, format, outputStream);
      final byte[] imageBytes = outputStream.toByteArray();
      final String base64 = Base64.getEncoder().encodeToString(imageBytes);
      return "data:image/" + format + ";base64," + base64;

    } catch (final Exception e) {
      IJ.log("Error encoding image to Base64: " + e.getMessage());
      return null;
    }
  }

  /**
   * Returns the raw base64 payload of a data-URI produced by
   * {@link #encodeImageToBase64(BufferedImage, String)} (i.e. drops the
   * leading {@code data:<mime>;base64,}). Values that are already bare base64
   * are returned unchanged, so this is safe to call on either form.
   */
  public static String extractBase64(final String dataUri) {
    if (dataUri == null) {
      return null;
    }
    final int comma = dataUri.indexOf(',');
    return comma < 0 ? dataUri : dataUri.substring(comma + 1);
  }
}
