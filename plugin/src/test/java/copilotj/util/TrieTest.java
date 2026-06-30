/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import copilotj.util.Trie.CharTrie;
import copilotj.util.Trie.SentenceTrie;
import copilotj.util.Trie.WordTrie;

/**
 * Unit tests for the pure-logic {@link Trie} data structure. {@link Trie} has no
 * ImageJ/AWT/runtime coupling, so it can be exercised directly.
 */
// Raw Trie.Map / Trie.SimpleMap are inner generic types; unchecked access is by design here.
@SuppressWarnings("rawtypes")
public class TrieTest {

  @Test
  public void charTrieInsertSearchAndPrefix() {
    final CharTrie trie = new CharTrie();
    trie.insert("apple");

    assertTrue(trie.search("apple"), "exact sentence should be found");
    assertFalse(trie.search("app"), "non-leaf prefix should not satisfy search");
    assertTrue(trie.startsWith("app"), "prefix should be detected");
    assertFalse(trie.startsWith("b"), "unknown prefix");
    assertFalse(new CharTrie().search("x"), "empty trie lookup handled");
  }

  @Test
  public void charTrieSharedPrefixDoesNotInterfere() {
    final CharTrie trie = new CharTrie();
    trie.insert("app");
    trie.insert("apple");

    assertTrue(trie.search("app"));
    assertTrue(trie.search("apple"));
    assertFalse(trie.search("ap"));
  }

  @Test
  public void duplicateInsertIsIdempotent() {
    final CharTrie trie = new CharTrie();
    trie.insert("abc");
    trie.insert("abc");

    assertTrue(trie.search("abc"));
  }

  @Test
  public void wordTrieSearchAndPrefix() {
    final WordTrie trie = new WordTrie();
    trie.insert(new String[] {"a", "b", "c"});

    assertTrue(trie.search(new String[] {"a", "b", "c"}));
    assertFalse(trie.search(new String[] {"a", "b"}));
    assertTrue(trie.startsWith(new String[] {"a", "b"}));
    assertFalse(trie.startsWith(new String[] {"x"}));
  }

  @Test
  public void wordTrieToMapWithFlattenCollapsesSingleChildChains() {
    // root -> a -> b -> {c (leaf), d (leaf)}; the single-child a->b chain is
    // flattened into the combined key "ab".
    final WordTrie trie = new WordTrie();
    trie.insert(new String[] {"a", "b", "c"});
    trie.insert(new String[] {"a", "b", "d"});

    final Trie.Map flattened = trie.toMapWithFlatten();
    assertNotNull(flattened.children);
    assertTrue(flattened.children.containsKey("ab"), "flattened single-child chain should produce key 'ab'");

    final Trie.Map ab = (Trie.Map) flattened.children.get("ab");
    assertNotNull(ab.children);
    assertTrue(ab.children.containsKey("c"));
    assertTrue(ab.children.containsKey("d"));
    assertTrue(((Trie.Map) ab.children.get("c")).isLeaf, "leaf node c should be marked");
  }

  @Test
  public void wordTrieToSimpleMapWithFlattenCollapsesSingleChildChains() {
    // nodeToSimpleMap flattens single-child chains via a while-loop (distinct from
    // toMap's single-edge flatten). This is the path that previously held a debug
    // println, so it is covered explicitly here.
    final WordTrie trie = new WordTrie();
    trie.insert(new String[] {"a", "b", "c"});
    trie.insert(new String[] {"a", "b", "d"});

    final Trie.SimpleMap flattened = trie.toSimpleMapWithFlatten();
    assertTrue(flattened.containsKey("ab"), "flattened single-child chain should produce key 'ab'");

    final Trie.SimpleMap ab = (Trie.SimpleMap) flattened.get("ab");
    assertTrue(ab.containsKey("c"));
    assertTrue(ab.containsKey("d"));
  }

  @Test
  public void sentenceTrieSplitsOnDelimiters() {
    // Delimiters stick to the preceding token: "a/b.c" -> ["a/", "b.", "c"].
    final SentenceTrie trie = new SentenceTrie(new char[] {'/', '.'});
    trie.insert("a/b.c");

    assertTrue(trie.search("a/b.c"), "full sentence reconstructs");
    assertFalse(trie.search("a/b"), "trailing-delimiter token is significant");
    assertTrue(trie.startsWith("a/"), "prefix token including its delimiter");
    assertTrue(trie.startsWith("a/b."), "prefix across two tokens");
    assertFalse(trie.startsWith("a/b"));
  }

  @Test
  public void toMapWithFlattenRejectsNullCombiner() {
    assertThrows(IllegalArgumentException.class, () -> new WordTrie().toMapWithFlatten(null));
  }

  @Test
  public void toSimpleMapWithFlattenRejectsNullCombiner() {
    assertThrows(IllegalArgumentException.class, () -> new WordTrie().toSimpleMapWithFlatten(null));
  }
}
