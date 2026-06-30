/**
 * SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package copilotj.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.fasterxml.jackson.annotation.JsonAnyGetter;

/**
 * A generic Trie implementation following the Python version's design.
 * Supports case sensitivity and different sentence types.
 */
public abstract class Trie<Word, Sentence> {
  public static class Node<Word> {
    public java.util.Map<Word, Node<Word>> children = new HashMap<>();
    public boolean isLeaf = false;
  }

  public class Map {
    public boolean isLeaf;
    public HashMap<Word, Map> children;
  }

  /**
   * Converts the trie to a nested Map structure, ignore the isLeaf field
   *
   * @return Map representation of the trie
   */
  @FunctionalInterface
  public interface KeyCombiner<Word> {
    Word combine(Word parentKey, Word childKey);
  }

  public class SimpleMap extends HashMap<Word, SimpleMap> {
    @JsonAnyGetter
    public java.util.Map<Word, SimpleMap> getMapContent() {
      return this;
    }
  }

  /**
   * Trie implementation for char sentences
   */
  public static class CharTrie extends Trie<Integer, String> {
    @Override
    protected Integer[] sentenceToWords(final String sentence) {

      return sentence.chars().boxed().toArray(Integer[]::new);
    }
  }

  /**
   * Trie implementation for word arrays
   */
  public static class WordTrie extends Trie<String, String[]> {
    public Map toMapWithFlatten() {
      return toMapWithFlatten((parentKey, childKey) -> parentKey + childKey);
    }

    public SimpleMap toSimpleMapWithFlatten() {
      return toSimpleMapWithFlatten((parentKey, childKey) -> parentKey + childKey);
    }

    @Override
    protected String[] sentenceToWords(final String[] sentence) {
      return sentence;
    }
  }

  /**
   * Trie implementation for delimited sentences with multiple delimiters
   */
  public static class SentenceTrie extends Trie<String, String> {
    private final Set<Character> delimiters;

    /**
     * Create a SentenceTrie with delimiter characters
     * 
     * @param delimiters Array of delimiter characters
     */
    public SentenceTrie(final char[] delimiters) {
      super();
      this.delimiters = new HashSet<>();
      for (final char c : delimiters) {
        this.delimiters.add(c);
      }
    }

    public Map toMapWithFlatten() {
      return toMapWithFlatten((parentKey, childKey) -> parentKey + childKey);
    }

    public SimpleMap toSimpleMapWithFlatten() {
      return toSimpleMapWithFlatten((parentKey, childKey) -> parentKey + childKey);
    }

    @Override
    protected String[] sentenceToWords(final String sentence) {
      final List<String> words = new ArrayList<>();
      final StringBuilder currentWord = new StringBuilder();

      for (final char c : sentence.toCharArray()) {
        currentWord.append(c);

        if (delimiters.contains(c)) {
          // Add current word and delimiter as separate entries
          if (currentWord.length() > 0) {
            words.add(currentWord.toString());
            currentWord.setLength(0);
          }
        }
      }

      // Add any remaining characters
      if (currentWord.length() > 0) {
        words.add(currentWord.toString());
      }

      return words.toArray(new String[0]);
    }
  }

  protected final Node<Word> root = new Node<>();

  /**
   * Inserts a sentence into the trie
   * 
   * @param sentence The sentence to insert
   */
  public void insert(final Sentence sentence) {
    Node<Word> current = root;
    for (final Word word : sentenceToWords(sentence)) {
      current = current.children.computeIfAbsent(word, k -> new Node<>());
    }
    current.isLeaf = true;
  }

  /**
   * Searches for a complete sentence in the trie
   * 
   * @param sentence The sentence to search for
   * @return true if the sentence exists, false otherwise
   */
  public boolean search(final Sentence sentence) {
    Node<Word> current = root;
    for (final Word word : sentenceToWords(sentence)) {
      if (!current.children.containsKey(word)) {
        return false;
      }
      current = current.children.get(word);
    }
    return current.isLeaf;
  }

  /**
   * Checks if any sentence starts with the given prefix
   *
   * @param prefix The prefix to check
   * @return true if any sentence starts with the prefix, false otherwise
   */
  public boolean startsWith(final Sentence prefix) {
    Node<Word> current = root;
    for (final Word word : sentenceToWords(prefix)) {
      if (!current.children.containsKey(word)) {
        return false;
      }
      current = current.children.get(word);
    }
    return true;
  }

  /**
   * Converts the trie to a nested Map structure
   *
   * @return Map representation of the trie
   */
  public Map toMap() {
    return nodeToMap(root, null);
  }

  /**
   * Converts the trie to a nested Map structure
   *
   * @return Map representation of the trie
   */
  public Map toMapWithFlatten(final KeyCombiner<Word> combiner) {
    if (combiner == null) {
      throw new IllegalArgumentException("combiner must not be null");
    }
    return nodeToMap(root, combiner);
  }

  public SimpleMap toSimpleMap() {
    return nodeToSimpleMap(root, null);
  }

  // TODO: move flatten to `Trie.flatten()`, or automatic flatten
  public SimpleMap toSimpleMapWithFlatten(final KeyCombiner<Word> combiner) {
    if (combiner == null) {
      throw new IllegalArgumentException("combiner must not be null");
    }

    return nodeToSimpleMap(root, combiner);
  }

  /**
   * Converts a sentence to its constituent words
   * 
   * @param sentence The sentence to convert
   * @return Array of words
   */
  protected abstract Word[] sentenceToWords(Sentence sentence);

  private Map nodeToMap(final Node<Word> node, final KeyCombiner<Word> combiner) {
    final Map map = new Map();
    map.isLeaf = node.isLeaf;
    if (node.children == null || node.children.isEmpty()) {
      return map;
    }

    map.children = new HashMap<>();
    for (final java.util.Map.Entry<Word, Node<Word>> entry : node.children.entrySet()) {
      final Word word = entry.getKey();
      final Node<Word> childNode = entry.getValue();

      if (combiner != null && childNode.children != null && childNode.children.size() == 1) {
        // Flatten single-child paths
        final java.util.Map.Entry<Word, Node<Word>> onlyChild = childNode.children.entrySet().iterator().next();
        final Word combinedKey = combiner.combine(word, onlyChild.getKey());
        map.children.put(combinedKey, nodeToMap(onlyChild.getValue(), combiner));
      } else {
        map.children.put(word, nodeToMap(childNode, combiner));
      }
    }
    return map;
  }

  private SimpleMap nodeToSimpleMap(final Node<Word> node, final KeyCombiner<Word> combiner) {
    if (node.children == null || node.children.isEmpty()) {
      return null;
    }

    final SimpleMap map = new SimpleMap();
    for (final java.util.Map.Entry<Word, Node<Word>> entry : node.children.entrySet()) {
      Word word = entry.getKey();
      Node<Word> childNode = entry.getValue();

      if (combiner != null) {
        // Flatten single-child paths
        while (childNode != null && childNode.children != null && childNode.children.size() == 1) {
          final java.util.Map.Entry<Word, Node<Word>> onlyChild = childNode.children.entrySet().iterator().next();
          word = combiner.combine(word, onlyChild.getKey());
          childNode = onlyChild.getValue();
        }
      }
      map.put(word, nodeToSimpleMap(childNode, combiner));
    }
    return map;
  }
}
