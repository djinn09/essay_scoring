"""A module for grading essays based on their semantic similarity to a set of example essays.

Example Essays
-------------

The example essays are stored in a directory specified by the `EXAMPLE_ESSAYS_DIR` environment variable.
Each example essay is a text file with a filename that indicates the topic of the essay.
The filename should be in the format `topic_name.txt`, where `topic_name` is the name of the topic.

Grading
-------

The grading process works as follows:

1. The user provides an essay to be graded.
2. The essay is preprocessed to remove punctuation and convert all words to lowercase.
3. The essay is split into sentences.
4. For each sentence, the semantic similarity to each example essay is computed using
the `semantic_similarity` function.
5. The average semantic similarity across all sentences is computed.
6. The average semantic similarity is returned as the grade for the essay.

Semantic Similarity
------------------

The semantic similarity between two pieces of text is computed using the following algorithm:

1. The text is split into sentences.
2. For each sentence, the semantic similarity to each sentence in the other text is computed using the
`sentence_similarity` function.
3. The average semantic similarity across all sentences is computed.
4. The average semantic similarity is returned as the semantic similarity between the two texts.

Sentence Similarity
-----------------

The sentence similarity between two sentences is computed using the following algorithm:

1. The sentences are split into words.
2. For each word in the first sentence, the similarity to each word in the second sentence is
computed using the `word_similarity` function.
3. The average similarity across all words is computed.
4. The average similarity is returned as the sentence similarity between the two sentences.

Word Similarity
-------------

The word similarity between two words is computed using the following algorithm:

1. The words are stemmed using the Porter stemmer.
2. The stemmed words are looked up in a dictionary to find their semantic similarity.
3. The semantic similarity is returned as the word similarity between the two words.

Stemming
---------

The Porter stemmer is used to stem words.
This is a simple algorithm that works by removing common suffixes from words.
For example, the word "running" would be stemmed to "run".

Dictionary
----------

The dictionary used to look up the semantic similarity between words is a text file specified by
the `DICTIONARY_FILE` environment variable.
The file should contain one entry per line, with each entry in the format `word1 word2 similarity`,
where `word1` and `word2` are the two words and `similarity` is their semantic similarity.

Environment Variables
--------------------

The following environment variables are used by this module:

* `EXAMPLE_ESSAYS_DIR`: The directory where the example essays are stored.
* `DICTIONARY_FILE`: The file containing the dictionary of word similarities.

"""
