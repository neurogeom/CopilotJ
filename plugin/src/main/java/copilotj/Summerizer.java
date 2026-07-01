/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Summarizer class to get summary of the context and environment
//
// References
// - https://github.com/scijava/scijava-plugins-commands/blob/57261c437778c462205ea8f458d4f99ab4ce452d/src/main/java/org/scijava/plugins/commands/debug/SystemInformation.java#L287
// - https://github.com/imagej/ImageJ/blob/03382be1a0b650010b077102b5e5b20eaa6a8967/ij/plugin/ImageInfo.java#L18

package copilotj;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.scijava.Context;
import org.scijava.log.LogService;
import org.scijava.plugin.PluginInfo;
import org.scijava.plugin.SciJavaPlugin;
import org.scijava.util.ClassUtils;

import ij.IJ;

import copilotj.util.Trie;

public class Summerizer {
  public static class EnvironmentSummary {
    public final String imagejHome;
    public final String javaHome;
    public final int javaVersion;
    public final Map<String, Trie.SentenceTrie.SimpleMap> plugins;

    public EnvironmentSummary(final String imagejHome, final String javaHome, final int javaVersion,
        final Map<String, Trie.SentenceTrie> plugins) {
      this.imagejHome = imagejHome;
      this.javaHome = javaHome;
      this.javaVersion = javaVersion;
      this.plugins = new LinkedHashMap<>();
      for (final Map.Entry<String, Trie.SentenceTrie> entry : plugins.entrySet()) {
        this.plugins.put(entry.getKey(), entry.getValue().toSimpleMapWithFlatten());
      }
    }
  }

  private final Context context;
  private final LogService log;

  public Summerizer(final Context context, final LogService log, final SnapshotManager contextManager) {
    this.context = context;
    this.log = log;
  }

  public EnvironmentSummary summariseEnvironment() {
    final String imagejHome = IJ.getDir("imagej");
    final String javaHome = System.getProperty("java.home");
    final int javaVersion = IJ.javaVersion();
    final Map<String, Trie.SentenceTrie> pluginsByType = listPlugins();
    return new EnvironmentSummary(imagejHome, javaHome, javaVersion, pluginsByType);
  }

  // --- Plugins --

  Map<String, Trie.SentenceTrie> listPlugins() {
    final ArrayList<Class<? extends SciJavaPlugin>> pluginTypes = listPluginTypes();
    final Map<String, Trie.SentenceTrie> result = new LinkedHashMap<>();

    // Build the tree structure
    for (final Class<? extends SciJavaPlugin> pluginType : pluginTypes) {
      final String typeName = pluginType.getName();
      final Trie.SentenceTrie pluginNames = listPluginsByType(pluginType);
      if (pluginNames != null) {
        result.put(typeName, pluginNames);
      }
    }

    return result;
  }

  ArrayList<Class<? extends SciJavaPlugin>> listPluginTypes() {
    // compute the set of known plugin types
    final List<PluginInfo<?>> plugins = context.getPluginIndex().getAll();
    final HashSet<Class<? extends SciJavaPlugin>> pluginTypeSet = new HashSet<>();

    for (final PluginInfo<?> plugin : plugins) {
      pluginTypeSet.add(plugin.getPluginType());
    }

    // convert to sorted list of plugin types
    final ArrayList<Class<? extends SciJavaPlugin>> pluginTypes = new ArrayList<>(pluginTypeSet);
    Collections.sort(pluginTypes, new Comparator<Class<?>>() {
      @Override
      public int compare(final Class<?> c1, final Class<?> c2) {
        return ClassUtils.compare(c1, c2);
      }
    });

    return pluginTypes;
  }

  <PT extends SciJavaPlugin> Trie.SentenceTrie listPluginsByType(final Class<PT> pluginType) {
    final Trie.SentenceTrie trie = new Trie.SentenceTrie(new char[] { '.', '$' });
    final List<PluginInfo<PT>> plugins = context.getPluginIndex().getPlugins(pluginType);

    // count the number of plugins whose type matches exactly (not sub-types)
    int count = 0;
    for (final PluginInfo<PT> plugin : plugins) {
      if (pluginType == plugin.getPluginType()) {
        count += 1;
        trie.insert(plugin.getClassName());
      }
    }
    log.debug("Found " + count + " plugins of type " + pluginType.getName());
    return count > 0 ? trie : null;
  }
}
