/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link JsonBase64ImageTruncator}, a pure regex/string utility
 * with no ImageJ coupling.
 */
public class JsonBase64ImageTruncatorTest {

  private static final String LONG = "AAAAAAAAAAAAAAAAAAAA"; // 20 chars

  @Test
  public void truncatesLongBase64Content() {
    final String json = "{\"img\":\"data:image/png;base64," + LONG + "\"}";
    // maxLength 16 -> keep max(0, 16-3)=13 chars, then append "...".
    final String expected = "{\"img\":\"data:image/png;base64,AAAAAAAAAAAAA...\"}";
    assertEquals(expected, new JsonBase64ImageTruncator(json, 16).toString());
  }

  @Test
  public void leavesShortContentUntouched() {
    final String content = "AAAAAAAAAA"; // 10 chars, <= 16
    final String json = "{\"img\":\"data:image/png;base64," + content + "\"}";
    assertEquals(json, new JsonBase64ImageTruncator(json, 16).toString());
  }

  @Test
  public void contentExactlyAtMaxLengthIsUntouched() {
    // Production truncates only when content.length() > maxLength (strict). Pin the
    // exact boundary so an off-by-one flip to >= would be caught.
    final String content = "AAAAAAAAAAAAAAAA"; // 16 chars == maxLength
    final String json = "{\"img\":\"data:image/png;base64," + content + "\"}";
    assertEquals(json, new JsonBase64ImageTruncator(json, 16).toString());
  }

  @Test
  public void noMatchReturnsOriginal() {
    final String json = "{\"msg\":\"hello world\"}";
    assertEquals(json, new JsonBase64ImageTruncator(json, 16).toString());
  }

  @Test
  public void truncatesEveryMatch() {
    final String json = "{\"a\":\"data:image/png;base64," + LONG + "\","
        + "\"b\":\"data:image/jpeg;base64," + LONG + "\"}";
    final String result = new JsonBase64ImageTruncator(json, 16).toString();
    assertEquals(2, countOccurrences(result, "..."));
  }

  @Test
  public void maxLengthBelowThreeCollapsesToEllipsis() {
    final String json = "{\"img\":\"data:image/png;base64," + LONG + "\"}";
    // max(0, 2-3)=0 -> substring(0,0) + "...".
    assertEquals("{\"img\":\"data:image/png;base64,...\"}",
        new JsonBase64ImageTruncator(json, 2).toString());
  }

  @Test
  public void withTitlePrependsTitle() {
    final String json = "{\"img\":\"data:image/png;base64," + LONG + "\"}";
    // WithTitle hardcodes maxLength = 16.
    final String result = new JsonBase64ImageTruncator.WithTitle("img", json).toString();
    assertEquals("img: {\"img\":\"data:image/png;base64,AAAAAAAAAAAAA...\"}", result);
  }

  @Test
  public void staticHelperMatchesInstance() {
    final String json = "{\"img\":\"data:image/png;base64," + LONG + "\"}";
    final String viaStatic = JsonBase64ImageTruncator.truncateJsonBase64(json, 16);
    final String viaInstance = new JsonBase64ImageTruncator(json, 16).toString();
    assertEquals(viaInstance, viaStatic);
  }

  @Test
  public void svgXmlMimeSubtypeIsNotMatchedAndPassesThrough() {
    // The regex matches only image/\w+ (single-token MIME), so a structured subtype
    // like image/svg+xml does NOT match and is left intact. Known limitation;
    // broadening the regex is tracked as a follow-up.
    final String json = "{\"img\":\"data:image/svg+xml;base64," + LONG + "\"}";
    assertEquals(json, new JsonBase64ImageTruncator(json, 4).toString());
  }

  private static int countOccurrences(final String haystack, final String needle) {
    int count = 0;
    int idx = 0;
    while ((idx = haystack.indexOf(needle, idx)) != -1) {
      count++;
      idx += needle.length();
    }
    return count;
  }
}
