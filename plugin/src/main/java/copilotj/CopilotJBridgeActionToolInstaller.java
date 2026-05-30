/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import ij.plugin.MacroInstaller;

class CopilotJBridgeActionToolInstaller {
  public static void install() {
    String macro = "macro \"CopilotJ Action Tool - " + getToolIcon() + "\" {\n" +
        "  run(\"CopilotJ\");\n" +
        "}\n";

    try {
      new MacroInstaller().install(macro);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private static String getToolIcon() {
    return generateIconCodeString(getToolIconString());
  }

  private static String getToolIconString() {
    return
    //////// 0123456789ABCDEF
    /* 0 */ " ###  ##### ####" +
    /* 1 */ "#   #   #      #" +
    /* 2 */ "#   #   #      #" +
    /* 3 */ "#       #      #" +
    /* 4 */ "#       #      #" +
    /* 5 */ "#       #      #" +
    /* 6 */ "#       #      #" +
    /* 7 */ "#       #      #" +
    /* 8 */ "#       #      #" +
    /* 9 */ "#       #      #" +
    /* A */ "#       #      #" +
    /* B */ "#       #      #" +
    /* C */ "#       #      #" +
    /* D */ "#   #   #      #" +
    /* E */ "#   #   #   #  #" +
    /* F */ " ###  #####  ## ";
  }

  private static String generateIconCodeString(String icon) {
    String[] positions = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "0" };

    String result = "C000";
    int x = 0;
    int y = 0;

    char empty = ' ';
    // DebugHelper.print(new Object(), "len: " + icon.length());
    for (int i = 0; i < icon.length(); i++) {
      // DebugHelper.print(new Object(), "|" + icon.charAt(i) + " == " + empty + "|");
      if (icon.charAt(i) != empty) {
        result = result.concat("D" + positions[x] + positions[y]);
      }

      x++;
      if (x > 15) {
        x = 0;
        y++;
      }
    }
    // DebugHelper.print(new Object(), result);
    return result;
  }
}
