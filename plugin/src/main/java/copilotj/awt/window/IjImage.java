/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.awt.window;

import java.awt.Rectangle;
import java.awt.Window;
import java.awt.image.BufferedImage;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import ij.ImagePlus;
import ij.gui.ImageWindow;
import ij.gui.Roi;
import ij.io.FileInfo;
import ij.measure.Calibration;
import ij.process.ImageProcessor;

import copilotj.ImagejListener;
import copilotj.awt.Action;
import copilotj.awt.WindowIdentifier;
import copilotj.util.FromTo;
import copilotj.util.IjImageHelper;

public class IjImage extends AbstractAwtWindow<ImageWindow> {
  public static class Provider implements AwtWindowProvider {
    @Override
    public AwtWindow tryCreate(final WindowIdentifier identifier, final Window window) throws InvalidWindowException {
      if (window instanceof ImageWindow) {
        final ImageWindow iw = (ImageWindow) window;
        if (iw.getImagePlus() == null) {
          throw new InvalidWindowException("ImagePlus is null");
        }

        return new IjImage(identifier, iw);
      }
      return null;
    }
  }

  public static class Difference extends AbstractAwtWindow.Difference {
    public final FromTo<String> imageType;
    public final FromTo<String> size;
    public final FromTo<String> path;

    public final FromTo<Integer> bitDepth;
    public final FromTo<Integer> stackSize;
    public final FromTo<Integer> channels;
    public final FromTo<Integer> slices;
    public final FromTo<Integer> frames;

    public final FromTo<Integer> width;
    public final FromTo<Integer> height;
    public final FromTo<Integer> depth;

    public final FromTo<Boolean> calibrated;
    public final FromTo<Double> calibratedWidth;
    public final FromTo<Double> calibratedHeight;
    public final FromTo<Double> calibratedDepth;

    public final FromTo<String> xUnit;
    public final FromTo<String> yUnit;
    public final FromTo<String> zUnit;

    public final FromTo<Double> xResolution;
    public final FromTo<Double> yResolution;

    public final FromTo<Double> zoomFactor;
    public final FromTo<Roi> roi;

    public final boolean imageUpdated;

    public Difference(final IjImage from, final IjImage to, final boolean imageUpdated) {
      super(from, to);

// @formatter:off
      this.imageType = !Objects.equals(from.imageType, to.imageType) ? new FromTo<>(from.imageType, to.imageType) : null;
      this.size =      !Objects.equals(from.size,      to.size)      ? new FromTo<>(from.size,      to.size)      : null;
      this.path =      !Objects.equals(from.path,      to.path)      ? new FromTo<>(from.path,      to.path)      : null;

      this.bitDepth =  !Objects.equals(from.bitDepth,  to.bitDepth)  ? new FromTo<>(from.bitDepth,  to.bitDepth)  : null;
      this.stackSize = !Objects.equals(from.stackSize, to.stackSize) ? new FromTo<>(from.stackSize, to.stackSize) : null;
      this.channels =  !Objects.equals(from.channels,  to.channels)  ? new FromTo<>(from.channels,  to.channels)  : null;
      this.slices =    !Objects.equals(from.slices,    to.slices)    ? new FromTo<>(from.slices,    to.slices)    : null;
      this.frames =    !Objects.equals(from.frames,    to.frames)    ? new FromTo<>(from.frames,    to.frames)    : null;

      this.width =  !Objects.equals(from.width, to.width)   ? new FromTo<>(from.width,  to.width) : null;
      this.height = !Objects.equals(from.height, to.height) ? new FromTo<>(from.height, to.height) : null;
      this.depth =  !Objects.equals(from.depth, to.depth)   ? new FromTo<>(from.depth,  to.depth) : null;

      this.calibrated       = !Objects.equals(from.calibrated,       to.calibrated)       ? new FromTo<>(from.calibrated,       to.calibrated)       : null;
      this.calibratedWidth  = !Objects.equals(from.calibratedWidth,  to.calibratedWidth)  ? new FromTo<>(from.calibratedWidth,  to.calibratedWidth)  : null;
      this.calibratedHeight = !Objects.equals(from.calibratedHeight, to.calibratedHeight) ? new FromTo<>(from.calibratedHeight, to.calibratedHeight) : null;
      this.calibratedDepth  = !Objects.equals(from.calibratedDepth,  to.calibratedDepth)  ? new FromTo<>(from.calibratedDepth,  to.calibratedDepth)  : null;

      this.xUnit = !Objects.equals(from.xUnit, to.xUnit) ? new FromTo<>(from.xUnit, to.xUnit) : null;
      this.yUnit = !Objects.equals(from.yUnit, to.yUnit) ? new FromTo<>(from.yUnit, to.yUnit) : null;
      this.zUnit = !Objects.equals(from.zUnit, to.zUnit) ? new FromTo<>(from.zUnit, to.zUnit) : null;

      this.xResolution = !Objects.equals(from.xResolution, to.xResolution) ? new FromTo<>(from.xResolution, to.xResolution) : null;
      this.yResolution = !Objects.equals(from.yResolution, to.yResolution) ? new FromTo<>(from.yResolution, to.yResolution) : null;

      this.zoomFactor = !Objects.equals(from.zoomFactor, to.zoomFactor) ? new FromTo<>(from.zoomFactor, to.zoomFactor) : null;
      this.roi = !Objects.equals(from.roi, to.roi) ? new FromTo<>(from.roi, to.roi) : null;
// @formatter:on

      this.imageUpdated = imageUpdated;
    }
  }

  public static class ImagePreview {
    public final String title;
    public final String image;
    public final int originalWidth;
    public final int originalHeight;
    public final int resizedWidth;
    public final int resizedHeight;
    public final float scaleFactor;

    public ImagePreview(final String title, final BufferedImage image, final int maxWidth, final int maxHeight) {
      // Downsize the image if it's too large
      final BufferedImage resizedImage = IjImageHelper.downsizeImage(image, maxWidth, maxHeight);
      // Convert the image to a Base64 string
      final String base64Image = IjImageHelper.encodeImageToBase64(resizedImage, "png");
      // Remove spaces and newlines
      final String cleanedBase64Image = base64Image.replaceAll("\\s+", "");

      this.title = title;
      this.image = cleanedBase64Image;
      this.originalWidth = image.getWidth();
      this.originalHeight = image.getHeight();
      this.resizedWidth = resizedImage.getWidth();
      this.resizedHeight = resizedImage.getHeight();
      this.scaleFactor = (float) resizedImage.getWidth() / image.getWidth();
    }
  }

  public interface Roi {
  }

  public static class RectangleRoi implements Roi {
    public final int x;
    public final int y;
    public final int width;
    public final int height;

    public RectangleRoi(final ij.gui.Roi roi) {
      if (roi.getType() != ij.gui.Roi.RECTANGLE) {
        throw new IllegalArgumentException("Roi is not a rectangle: " + roi);
      }

      final Rectangle bounds = roi.getBounds();
      this.x = bounds.x;
      this.y = bounds.y;
      this.width = bounds.width;
      this.height = bounds.height;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) {
        return true;
      } else if (o == null || getClass() != o.getClass()) {
        return false;
      }
      final RectangleRoi roiInfo = (RectangleRoi) o;
      return x == roiInfo.x && y == roiInfo.y && width == roiInfo.width && height == roiInfo.height;
    }

    @Override
    public int hashCode() {
      return Objects.hash(x, y, width, height);
    }
  }

  public static class DescribedRoi implements Roi {
    public final String description;

    public DescribedRoi(final ij.gui.Roi roi) {
      final String type = roi.getTypeAsString();
      final Rectangle bounds = roi.getBounds();

      switch (roi.getType()) {
        case ij.gui.Roi.OVAL:
          this.description = String.format("Oval selection at [%d, %d] with size %dx%d", bounds.x, bounds.y,
              bounds.width, bounds.height);
          break;

        case ij.gui.Roi.POLYGON:
        case ij.gui.Roi.FREEROI: // Freehand selection
          this.description = String.format("%s selection with %d points", type, roi.getContainedPoints().length);
          break;

        case ij.gui.Roi.LINE:
        case ij.gui.Roi.POLYLINE:
        case ij.gui.Roi.FREELINE:
          this.description = String.format("%s with %d points", type, roi.getContainedPoints().length);
          break;

        case ij.gui.Roi.POINT:
          final int n = roi.getContainedPoints().length;
          this.description = n == 1
              ? String.format("Point selection at [%d, %d]", bounds.x, bounds.y)
              : String.format("Point selection with %d points", n);

          break;

        default:
          // For other types like TRACED_ROI, ANGLE, COMPOSITE, or unknown types,
          // use the default toString() method.
          this.description = roi.toString();
      }
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) {
        return true;
      } else if (o == null || getClass() != o.getClass()) {
        return false;
      }
      final DescribedRoi that = (DescribedRoi) o;
      return Objects.equals(description, that.description);
    }

    @Override
    public int hashCode() {
      return Objects.hash(description);
    }
  }

  public static class ImagePreviewWithInfo extends ImagePreview {
    public static class Histogram {
      public final int binCount;
      public final int[] bins;
      public final double minGray;
      public final double maxGray;

      public Histogram(final ImageProcessor ip, final int binCount) {
        this.binCount = binCount;
        this.bins = ip.getHistogram(binCount);
        this.minGray = ip.getMin();
        this.maxGray = ip.getMax();
      }
    }

    public final IjImage info;

    public final Histogram histogram;

    private ImagePreviewWithInfo(final WindowIdentifier identifier, final ImagePlus imp, final int maxWidth,
        final int maxHeight) {
      super(imp.getTitle(), imp.getBufferedImage(), maxWidth, maxHeight);
      this.info = new IjImage(identifier, imp.getWindow());

      final ImageProcessor ip = imp.getProcessor();
      this.histogram = new Histogram(ip, 32);
    }
  }

  public static ImagePreviewWithInfo captureImage(final WindowIdentifier identifier, final ImagePlus imp,
      final int maxWidth, final int maxHeight) {
    return new ImagePreviewWithInfo(identifier, imp, maxWidth, maxHeight);
  }

  public static ImagePreviewWithInfo captureImage(final WindowIdentifier identifier, final ImagePlus imp) {
    return captureImage(identifier, imp, 512, 512);
  }

  private static final String TYPE = "ij.Image";

  public final String title;
  public final String imageType;
  public final String size;
  public final String path;

  public final int bitDepth;
  public final int stackSize;
  public final int channels;
  public final int slices;
  public final int frames;

  public final int width;
  public final int height;
  public final int depth;

  public final boolean calibrated;
  public final double calibratedWidth;
  public final double calibratedHeight;
  public final double calibratedDepth;

  public final String xUnit;
  public final String yUnit;
  public final String zUnit;

  public final double xResolution;
  public final double yResolution;

  public final double zoomFactor;
  public final Roi roi;

  public IjImage(final WindowIdentifier identifier, final ImageWindow w) {
    super(identifier, w, TYPE);

    final ImagePlus imp = w.getImagePlus();
    // this.id = imp.getID(); // NOTE: we dont use Ij window id
    this.title = w.getImagePlus().getTitle();
    this.imageType = getImageType(imp);
    this.size = ImageWindow.getImageSize(imp);

    final FileInfo fi = imp.getOriginalFileInfo();
    String path = null;
    if (fi != null) {
      if (fi.url != null && !Objects.equals(fi.url, "")) {
        path += "URL: " + fi.url + "\n";
      } else {
        String defaultDir = (fi.directory == null || fi.directory.length() == 0) ? System.getProperty("user.dir")
            : "";
        if (defaultDir.length() > 0) {
          defaultDir = defaultDir.replaceAll("\\\\", "/");
          defaultDir += "/";
        }
        path = defaultDir + fi.getFilePath();
      }
    }
    this.path = path;

    this.bitDepth = imp.getBitDepth();
    this.stackSize = imp.getStackSize();
    this.channels = imp.getNChannels();
    this.slices = imp.getNSlices();
    this.frames = imp.getNFrames();

    this.width = imp.getWidth();
    this.height = imp.getHeight();
    this.depth = slices > 1 ? slices : 0;

    final Calibration cal = imp.getCalibration();
    this.calibrated = cal.scaled();
    this.xUnit = cal.getXUnit();
    this.yUnit = cal.getYUnit();
    this.zUnit = cal.getZUnit();

    this.calibratedWidth = imp.getWidth() * cal.pixelWidth;
    this.calibratedHeight = imp.getHeight() * cal.pixelHeight;
    this.calibratedDepth = slices > 1 ? slices * cal.pixelDepth : 0;

    this.xResolution = 1.0 / cal.pixelWidth;
    this.yResolution = 1.0 / cal.pixelHeight;

    this.zoomFactor = imp.getCanvas().getMagnification();

    final ij.gui.Roi roi = imp.getRoi();
    if (roi != null) {
      this.roi = roi.getType() == ij.gui.Roi.RECTANGLE ? new RectangleRoi(roi) : new DescribedRoi(roi);
    } else {
      this.roi = null;
    }
  }

  public boolean equals(final IjImage other) {
    return this.id == other.id
        && Objects.equals(this.title, other.title)
        && Objects.equals(this.type, other.type)
        && Objects.equals(this.size, other.size)
        && Objects.equals(this.path, other.path)

        && this.bitDepth == other.bitDepth
        && this.stackSize == other.stackSize
        && this.channels == other.channels
        && this.slices == other.slices
        && this.frames == other.frames

        && this.width == other.width
        && this.height == other.height
        && this.depth == other.depth

        && this.calibrated == other.calibrated
        && this.calibratedWidth == other.calibratedWidth
        && this.calibratedHeight == other.calibratedHeight
        && this.calibratedDepth == other.calibratedDepth

        && Objects.equals(this.xUnit, other.xUnit)
        && Objects.equals(this.yUnit, other.yUnit)
        && Objects.equals(this.zUnit, other.zUnit)

        && this.xResolution == other.xResolution
        && this.yResolution == other.yResolution
        && this.zoomFactor == other.zoomFactor
        && Objects.equals(this.roi, other.roi);
  }

  @Override
  public AwtWindow.Difference compare(final AwtWindow from, final ImagejListener.HistoryResponse history) {
    if (!(from instanceof IjImage)) {
      return super.compare(from, history);
    }

    final boolean imageUpdated = history.isImageUpdated(this.title);
    if (!imageUpdated && this.equals(from)) {
      return null;
    }

    return new Difference((IjImage) from, this, imageUpdated);
  }

  public List<Action> getActions() {
    final Action captureImage = Action
        .builder(TYPE + ".capture", "Capture", "Capture image in current window")
        .build();
    return Collections.singletonList(captureImage);
  }

  public Object runAction(final List<String> path, final String type, final List<Object> parameters) {
    if (!this.isActivate()) {
      throw new IllegalStateException("Window is not active");
    } else if (path.size() != 0) {
      throw new IllegalArgumentException("Path must be empty for IjImage");
    }

    switch (type) {
      case TYPE + ".capture":
        if (parameters != null && parameters.size() != 0) {
          throw new IllegalArgumentException("Action 'Capture' does not accept any parameters. Found: " + parameters);
        }

        return captureImage(this.identifier, this.component.getImagePlus());

      default:
        return super.runAction(path, type, parameters);
    }
  }

  private String getImageType(final ImagePlus imp) {
    switch (imp.getType()) {
      case ImagePlus.GRAY8:
        return "8-bit grayscale (unsigned)";
      case ImagePlus.GRAY16:
        return "16-bit grayscale (unsigned)";
      case ImagePlus.GRAY32:
        return "32-bit grayscale (float, " + getImageLutInfo(imp) + ")";
      case ImagePlus.COLOR_RGB:
        return "32-bit RGC color";
      case ImagePlus.COLOR_256:
        return "8-bit indexed color";
      default:
        return "Unknown";
    }
  }

  private String getImageLutInfo(final ImagePlus imp) {
    final String lut = imp.getProcessor().isColorLut() ? "color LUT" : "grayscale LUT";
    return imp.isInvertedLut() ? "inverting " + lut : lut;
  }
}
