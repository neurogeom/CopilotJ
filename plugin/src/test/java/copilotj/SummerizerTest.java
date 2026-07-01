/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.scijava.Context;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.PluginIndex;
import org.scijava.plugin.PluginInfo;
import org.scijava.plugin.SciJavaPlugin;
import org.scijava.service.Service;

import copilotj.util.Trie;

/**
 * Tests {@link Summerizer}'s plugin-listing logic with a mocked SciJava
 * {@link Context}. {@code summariseEnvironment()} (which calls {@code ij.IJ}
 * statics) is covered by Layer C; here we exercise the pure, Context-driven
 * helpers widened to package-private.
 */
public class SummerizerTest {

  private final Context context = mock(Context.class);
  private final LogService log = mock(LogService.class);
  private final PluginIndex pluginIndex = mock(PluginIndex.class);

  private Summerizer newSummerizer() {
    when(context.getPluginIndex()).thenReturn(pluginIndex);
    return new Summerizer(context, log, null); // contextManager param is unused
  }

  /** Build a stub PluginInfo whose getPluginType()/getClassName() are fixed. */
  private PluginInfo<?> info(final Class<? extends SciJavaPlugin> type, final String className) {
    final PluginInfo<?> info = mock(PluginInfo.class);
    doReturn(type).when(info).getPluginType(); // generic return → doReturn avoids variance issues
    when(info.getClassName()).thenReturn(className);
    return info;
  }

  @Test
  public void listPluginTypesDistinctAndSorted() {
    final PluginInfo<?> cmd1 = info(Command.class, "org.x.CommandA");
    final PluginInfo<?> cmd2 = info(Command.class, "org.x.CommandB");
    final PluginInfo<?> svc1 = info(Service.class, "org.x.ServiceA");
    when(pluginIndex.getAll()).thenReturn(Arrays.asList(cmd1, cmd2, svc1));
    final Summerizer s = newSummerizer();

    final ArrayList<Class<? extends SciJavaPlugin>> types = s.listPluginTypes();
    assertEquals(2, types.size(), "distinct types only");
    assertTrue(types.contains(Command.class));
    assertTrue(types.contains(Service.class));
    // sorted by ClassUtils.compare (lexicographic by class name)
    assertTrue(types.get(0).getName().compareTo(types.get(1).getName()) <= 0,
        "expected sorted by name, got " + types.get(0).getName() + " then " + types.get(1).getName());
  }

  @Test
  public void listPluginsByTypeReturnsTrieForExactMatches() {
    final PluginInfo<?> cmd1 = info(Command.class, "org.x.CommandA");
    final PluginInfo<?> cmd2 = info(Command.class, "org.x.CommandB");
    doReturn(Arrays.asList(cmd1, cmd2)).when(pluginIndex).getPlugins(Command.class);
    final Summerizer s = newSummerizer();

    final Trie.SentenceTrie trie = s.listPluginsByType(Command.class);
    assertNotNull(trie);
    assertNotNull(trie.toSimpleMapWithFlatten(), "two inserted class names yield a non-empty flatten");
  }

  @Test
  public void listPluginsByTypeReturnsNullWhenNoExactMatch() {
    // cmd1 is a Command, but we ask for Service plugins → exact-match count 0 → null.
    final PluginInfo<?> cmd1 = info(Command.class, "org.x.CommandA");
    doReturn(Arrays.asList(cmd1)).when(pluginIndex).getPlugins(Service.class);
    final Summerizer s = newSummerizer();

    assertNull(s.listPluginsByType(Service.class));
  }

  @Test
  public void listPluginsOmitsTypesWithNoExactMatches() {
    // getAll has Command + Service types; Service's getPlugins returns only a Command
    // (no exact match) → listPluginsByType returns null → Service omitted from the map.
    final PluginInfo<?> cmd1 = info(Command.class, "org.x.CommandA");
    final PluginInfo<?> cmd2 = info(Command.class, "org.x.CommandB");
    final PluginInfo<?> svc1 = info(Service.class, "org.x.ServiceA");
    when(pluginIndex.getAll()).thenReturn(Arrays.asList(cmd1, cmd2, svc1));
    doReturn(Arrays.asList(cmd1, cmd2)).when(pluginIndex).getPlugins(Command.class);
    doReturn(Arrays.asList(cmd1)).when(pluginIndex).getPlugins(Service.class);
    final Summerizer s = newSummerizer();

    final Map<String, Trie.SentenceTrie> map = s.listPlugins();
    assertEquals(1, map.size(), "Service omitted (no exact-match plugins)");
    assertTrue(map.containsKey(Command.class.getName()));
    assertFalse(map.containsKey(Service.class.getName()));
  }
}
