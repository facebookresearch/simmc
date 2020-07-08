"""Loads vocabulary and performs additional text processing.

Author(s): Satwik Kottur
"""
#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals

import copy


class Vocabulary:
    def __init__(self, vocabulary_path=None, immutable=False, verbose=True):
        """Initialize the vocabulary object given a path, else empty object.

    Args:
      vocabulary_path: List of words in a text file, one in each line.
      immutable: Once initialized, no new words can be added.
    """
        self.immutable = immutable
        self.verbose = verbose
        # Read file else create empty object.
        if vocabulary_path is not None:
            if verbose:
                print("Reading vocabulary: {0}...".format(vocabulary_path), end="")
            with open(vocabulary_path, "r") as file_id:
                self._words = [ii.strip() for ii in file_id.readlines()]
            if verbose:
                print("done")
            # Setup rest of the object.
            self._setup_vocabulary()
        else:
            if verbose:
                print("Initializing empty vocabulary object..")

    def __contains__(self, key):
        """Check if a word is contained in a vocabulary.
      """
        return key in self._words

    def _setup_vocabulary(self):
        """Sets up internal dictionaries.
    """
        # Check whether <unk>,<start>,<end> and <pad> are part of the word list.
        # Else add them.
        for special_word in ["<unk>", "<start>", "<end>", "<pad>"]:
            if special_word not in self._words:
                if not self.immutable:
                    self._words.append(special_word)
                    if self.verbose:
                        print("Adding new word to vocabulary: {}".format(special_word))
                else:
                    if self.verbose:
                        print("Immutable, cannot add missing {}".format(special_word))
        # Create word_index and word_string dictionaries.
        self.word_index = {word: index for index, word in enumerate(self._words)}
        self.word_string = {index: word for word, index in self.word_index.items()}
        if self.verbose:
            print("Vocabulary size updated: {0}".format(len(self.word_index)))

    def add_new_word(self, *new_words):
        """Adds new words to an existing vocabulary object.

    Args:
      *new_words: List of new word(s) to be added.
    """
        raise NotImplementedError

    def word(self, index):
        """Returns the word given the index.

    Args:
      index: Index of the word

    Returns:
      Word string for the given index.
    """
        assert index in self.word_string, "{0} missing in vocabulary!".format(index)
        return self.word_string[index]

    def index(self, word, unk_default=False):
        """Returns the index given the word.

    Args:
      word: Word string.

    Returns:
      Index for the given word string.
    """
        if not unk_default:
            assert word in self.word_index, "{0} missing in vocabulary!".format(word)
            return self.word_index[word]
        else:
            return self.word_index.get(word, self.word_index["<unk>"])

    def set_vocabulary_state(self, state):
        """Given a state (list of words), setup the vocabulary object state.

    Args:
      state: List of words
    """
        self._words = copy.deepcopy(state)
        self._setup_vocabulary()

    def get_vocabulary_state(self):
        """Returns the vocabulary state (deepcopy).

    Returns:
      Deepcopy of list of words.
    """
        return copy.deepcopy(self._words)

    def get_tensor_string(self, tensor):
        """Converts a tensor into a string after decoding it using vocabulary.
    """
        pad_token = self.index("<pad>")
        string = " ".join(
            [self.word(int(ii)) for ii in tensor.squeeze() if ii != pad_token]
        )
        return string

    @property
    def vocab_size(self):
        return len(self._words)
