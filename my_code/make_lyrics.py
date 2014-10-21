
"""


PoemGenerator(LanguageDefinition)
.GetLine(num_syllables, rhyming_word=None, attempts=10000)
.GetLineLogOdds(line or word_list)



LanguageDefinition(TextDataset, RhymingDictionary)
Get all lines from dataset
Compile into data structures
word_pair_freq[word1][word2] = freq
next_word[word1] = [(word2,prob), (word2,prob), ...]
.GetNextWord(word1, index) --> (word2, probability) or None
.GetNextWord(word1) --> word2   (random)
.GetPairLogOdds(word1, word2, default=None) --> odds




TextDataset
word_set = set([])  --> words
line_list = []  --> (src_index, line) pairs
source_list = list of filenames
.AddFile
.GetWords --> return a list of all words (sorted)
.GetLine(index)


RhymingDictionary
rhymes: word --> wordlist

.AddWord
   Only insert if it's not already known
.AddWords
   Keep ArrayList of words waiting to be parsed
.Compile
   Run binary and pass all waiting words through it
   Parse output: get rhymes and syllables for each word
   clear waiting list
.GetRhymes(word) --> wordlist
.GetSyllables(word or line) - returns sum of syllables of words in line
._SaveRhymeList(word_syllables_pairs): all these words rhyme  (word, syllables)




Code
------
text_dataset = TextDataset()
text_dataset.AddFile()

rhyming_dict = RhymingDictionary()
rhyming_dict.AddWords(text_dataset.GetWords())
rhyming_dict.Compile()

lang_def = LanguageDefinitiontext_dataset, rhyming_dict)




Misc
-------

Parse
Rhyming Dictionary 0.9 - interactive mode
 - exits on an empty line
 - type . to toggle between rhyme and syllable searches
 - type , to toggle between merged and non-merged results
 - type ? for help
RHYME> Finding perfect rhymes for app...
1: app, cap, capp, chap, chapp, clap, clapp, crap, dapp, flap, frap, gap, gapp
   hap, happ, happe, jap, kapp, klapp, knapp, krapp, lap, lapp, lappe, map, mapp
   nap, napp, pap, papp, rap, rapp, rappe, sap, sapp, schapp, schnapp, scrap
   shap, shapp, slap, snap, snapp, stapp, strap, tap, tapp, tappe, trap, trapp
   trappe, wrap, yap, yapp, zap, zapp

2: entrap, giap, mayhap(2), recap(2), unwrap

RHYME> Finding perfect rhymes for apple...
2: appel, appell, apple, cappel, chapel, chappel, chappell, chapple, grapple
   happel, kappel, mapel, schappell, shappell, snapple, stapel




TODO
-----
Current file input relies on sentences being on one line (like lyrics), but does
not match the input format of books, where sentences span several lines. Will need
to specify the filetype (simple/multiline) with the filename. Write code to handle
the non-lyrics format.
- A blank line is the end of a sentence
- Ignore UPPERCASE words (and sentences)
- Sentences with commas are non-trivial to parse.

Handle words with different numbers of syllables: e.g. every = 2-3
Handle trigrams (bigrams --> unigram)
Handle making of rhymes

debug:
- What are the words that exist in the dataset, but don't have ant rhyme matches?
These are also words whose number of syllables is unknown.

"""

import math
import os
import string
import random
import subprocess


ALLOWED_CHARS = string.ascii_lowercase + "'"

def CleanWord(word):
  cleaned = "".join([c for c in word.lower() if c in ALLOWED_CHARS])
  if not cleaned: return None
  if cleaned[0] not in string.ascii_lowercase: return None
  return cleaned


"""
PoemGenerator(LanguageDefinition)
.GetLine(num_syllables, rhyming_word=None, attempts=10000)
.GetLineLogProb(line or word_list)
"""
class PoemGenerator:
  def __init__(self, language_def):
    self.language_def = language_def
    self.debug = False

  def GetLine(self, num_syllables, rhyming_word=None, attempts=100000, allow_guess=False,
              last_word_must_have_rhymes=False):
    assert attempts > 0
    # Find possible rhyme endings
    word_rhymes = self.language_def.GetRhymes(rhyming_word) if rhyming_word else []
    # pre-compute syllables
    word_rhymes_info = [(word, self.language_def.GetSyllables(word, allow_guess=allow_guess))
                        for word in word_rhymes if word != rhyming_word]
    assert not rhyming_word or word_rhymes_info
    best = None
    best_score = float('-inf')
    curr = []
    for attempt in range(attempts):
      # Keep adding words until we've reached 'num_syllables'
      while self.language_def.GetSyllables(curr, allow_guess=allow_guess) < num_syllables:
        syllable_count = self.language_def.GetSyllables(curr, allow_guess=allow_guess)
        assert syllable_count is not None
        if not curr:
          preword = "."
        else:
          preword = curr[-1]
        # See if we can finish on a rhyming word, if so then pick a random one
        remaining_syllables = num_syllables - syllable_count
        rhyming_finish_words = [word for (word,sy) in word_rhymes_info if sy == remaining_syllables]
        if rhyming_finish_words:
          curr.append(rhyming_finish_words[random.randint(0, len(rhyming_finish_words)-1)])
        else:
          # Otherwise, just add another word
          next_word_info = self.language_def.GetNextWord(preword)
          if next_word_info is None: break
          next_word, prob = next_word_info
          curr.append(next_word)
          if curr[-1] in [".", None]: break
      if self.language_def.GetSyllables(curr, allow_guess) == num_syllables:
        # Assess line score
        if rhyming_word and not rhyming_finish_words:
          # We didn't finish on a rhyme
          score = -float('inf')
        elif last_word_must_have_rhymes and len(self.language_def.GetRhymes(curr[-1])) < 2:
          score = -float('inf')
        else:
          score = self._GetLineLogProb(curr)
        if score > best_score:
          best_score = score
          best = " ".join(curr)
          if self.debug:
            print("S=%0.9g : %s" % (score, best))
      # else:
      #   print("Wrong number of syllables: ", curr, self.language_def.GetSyllables(curr))

      # Clear line? (remove some words)
      curr = curr[:random.randint(0, len(curr))]

    return best

  def _GetLineLogProb(self, line):
    if type(line) is str:
      words = line.split()
    else:
      assert type(line) in [list, tuple]
      words = line
    score = 0.0
    default_logprob = -100
    for (i,w) in enumerate(words):
      if i == 0:
        score += self.language_def.GetPairLogProb(".", w, default=default_logprob)
      else:
        score += self.language_def.GetPairLogProb(words[i-1], w, default=default_logprob)
    score += self.language_def.GetPairLogProb(words[len(words)-1], ".", default=default_logprob)
    return score



"""
LanguageDefinition(TextDataset, RhymingDictionary)
Get all lines from dataset
Compile into data structures
word_pair_freq[word1][word2] = freq
next_word[word1] = [(word2,prob), (word2,prob), ...]
.GetNextWord(word1, index) --> (word2, probability) or None
.GetNextWord(word1, index=None) --> word2   (random)
.GetPairLogProb(word1, word2, default=None) --> Prob

"""
class LanguageDefinition:
  def __init__(self, text_dataset, rhyming_dict):
    self.rhyming_dict = rhyming_dict
    self.word_pair_freq = {}
    self.word_freq = {}
    self.next_word = {}

    print("Compiling LanguageDefinition")

    assert text_dataset.GetLineCount() > 0

    # Parse text dataset
    index = 0
    while text_dataset.GetLine(index) is not None:
      line = text_dataset.GetLine(index)
      index += 1
      # Note: line can be empty, since it 'cleans' the words
      # e.g. "92" will become ""
      if not line: continue
      words = line.split()
      for (i, w) in enumerate(words):
        if i == 0:
          self._Insert(".", w)
        else:
          self._Insert(words[i-1], w)
      if words:
        self._Insert(words[-1], ".")

    print("Parsed %d lines out of %d" % (index, text_dataset.GetLineCount()))
    self._CompileNextWord();

  def GetRhymes(self, word):
    return self.rhyming_dict.GetRhymes(word)

  def GetSyllables(self, line, allow_guess=False):
    return self.rhyming_dict.GetSyllables(line, allow_guess=allow_guess)

  def GetNextWord(self, word1, index=None):
    if word1 not in self.next_word: return None
    if index is None:
      return self._PickNextWord(word1)
    if index >= len(self.next_word[word1]): return None
    return self.next_word[word1][index]

  def GetPairLogProb(self, word1, word2, default=None):
    if word1 not in self.word_pair_freq: return default
    if word2 not in self.word_pair_freq[word1]: return default
    prob = self.word_pair_freq[word1][word2] / self.word_freq[word1]
    return math.log(prob)


  def _PickNextWord(self, word1):
    i = random.randint(0, len(self.next_word[word1])-1)
    return self.next_word[word1][i]

  def _PickNextWordUsingProbability(self, word1):
    r = random.random()
    for (w, p) in self.next_word[word1]:
      if p > r: return (w, p)
      r = r - p
    return (w, p)

  def _Insert(self, word1, word2):
    if word1 not in self.word_pair_freq:
      self.word_pair_freq[word1] = {}
    if word2 not in self.word_pair_freq[word1]:
      self.word_pair_freq[word1][word2] = 1
    else:
      self.word_pair_freq[word1][word2] += 1

  def _CompileNextWord(self):
    self.next_word = {}
    for word in self.word_pair_freq:
      flist = sorted([(freq, word2) for (word2, freq) in self.word_pair_freq[word].items()],
                     reverse=True)
      total_count = float(sum([freq for (freq, word2) in flist]))
      self.word_freq[word] = total_count
      self.next_word[word] = [(word2, freq/total_count) for (freq, word2) in flist]



"""
TextFileParser: Base class 
Subclasses should implement GetLine()
"""
class TextFileParser:
  def __init__(self, filename):
    self.filename = filename
    self.fp = open(filename, "rt")

  def __iter__(self):
    for line in self.GetLines():
      yield line

class LyricsParser(TextFileParser):
  def __init__(self, filename):
    TextFileParser.__init__(self, filename)

  def GetLines(self):
    for line in self.fp:
      line = line.strip()
      if not line: continue
      yield line


def FindFirst(s, targets):
  # Returns target, index
  best = len(s)
  best_target = None
  for t in targets:
    i = s.find(t)
    if (i > -1) and (i < best):
      best = i
      best_target = t
  return best_target, best


# TODO:
# all-too-human  --> should not be alltoohuman
class BookParser(TextFileParser):
  end_of_sentence_indicators = [".", "!", "?", ";", ":", "--"]

  def __init__(self, filename):
    TextFileParser.__init__(self, filename)
    self.skipped = 0
    self.total = 0


  def _ParseSentence(self, sentence):
    for c in [".", "!", "?", ";", ":"]:
      assert c not in sentence
    sentence.replace("-", " ")
    # Exclude sentences with , unless it precedes "and" (michael, and ian)
    # -- compute the % of dropped sentences
    # TODO: ignore sentences that are all uppercase (or 90%)
    #print("Parse: %s" % sentence)
    self.total += 1
    if "," in sentence:
      self.skipped += 1
    elif len(sentence.strip().split()) > 1:
      yield sentence
    else:
      self.skipped += 1

  def GetLines(self):
    partial_line = ""
    for line in self.fp:
      line = line.strip()
      # line = line.replace("--", " ")
      if not line:
        # Blank line indicates end-of-sentence
        if partial_line:
          yield partial_line
          partial_line = ""
        continue

      remainder = line
      while remainder:
        # A sentence ends in . or ; ? !
        found, index = FindFirst(remainder, BookParser.end_of_sentence_indicators)
        if found:
          s1 = partial_line + " " + remainder[:index]
          partial_line = ""
          remainder = remainder[(index+len(found)):]

          for s in self._ParseSentence(s1):
            yield s
        else:
          partial_line += " " + remainder
          remainder = ""

    print("partial_line = [%s]" % partial_line)
    if partial_line:
      yield partial_line
      partial_line = ""
    print("%s: Skipped %d/%d" % (self.filename, self.skipped, self.total))


"""

TextDataset
word_set = set([])  --> words
line_list = []  --> (src_index, line) pairs
source_list = list of filenames
.AddFile
.GetWords --> return a list of all words (sorted)
.GetLine(index)
"""
class TextDataset:
  def __init__(self):
    self.word_set = set([])
    self.line_list = []
    self.src_list = []

  def AddFile(self, line_source, source_description=None):
    index = len(self.src_list)
    self.src_list.append(source_description)
    for line in line_source:
      line = line.strip()
      if not line: continue
      self.line_list.append((index, line))
      for word in line.split():
        if CleanWord(word):
          self.word_set.add(CleanWord(word))

  # def AddFile_old(self, filename, parser):
  #   index = len(self.src_list)
  #   self.src_list.append(filename)
  #   with open(filename, 'r') as fp:
  #     for line in fp:
  #       line = line.strip()
  #       if not line: continue
  #       self.line_list.append((index, line))
  #       for word in line.split():
  #         if CleanWord(word):
  #           self.word_set.add(CleanWord(word))

  def GetRawLine(self, index):
    if index >= len(self.line_list):
      return None
    return self.line_list[index][1]

  def GetLineCount(self):
    return len(self.line_list)

  def GetLine(self, index):
    raw_line = self.GetRawLine(index)
    if not raw_line: return raw_line
    clean_line = " ".join([CleanWord(w) for w in raw_line.split() if CleanWord(w)])
    # if not clean_line:
    #   print("Warning: Empty clean line: %s" % raw_line)
    return clean_line

  def GetWords(self):
    return tuple(sorted(self.word_set))




"""

RhymingDictionary
rhymes: word --> wordlist

.AddWord
   Only insert if it's not already known
.AddWords
   Keep ArrayList of words waiting to be parsed
.Compile
   Run binary and pass all waiting words through it
   Parse output: get rhymes and syllables for each word
   clear waiting list
.GetRhymes(word) --> wordlist
.GetSyllables(word or line) - returns sum of syllables of words in line
._SaveRhymeList(word_syllables_pairs): all these words rhyme  (word, syllables)
"""
class RhymingDictionary:
  def __init__(self):
    # Map word to list of rhyming words
    self.rhymes = {}
    self.num_syllables = {}
    self.word_queue = []
    self.syllables_per_letter = None

  def AddWord(self, word):
    if word not in self.rhymes:
      self.word_queue.append(word)

  def AddWords(self, words):
    words = [w for w in words if w not in self.rhymes]
    self.word_queue = self.word_queue + words

  def GetRhymes(self, word):
    if word not in self.rhymes: return ()
    return tuple(self.rhymes[word])

  def GetSyllables(self, line, allow_guess=False):
    if type(line) == str:
      word_list = line.split()
    else:
      assert type(line) in [tuple, list]
      word_list = line
    count = 0
    for word in word_list:
      if word in self.num_syllables:
        count += self.num_syllables[word]
      else:
        if allow_guess:
          guess = max(1, int(round(len(word) * self.syllables_per_letter)))
          count += guess
        else:
          # Return None: unable to count syllables
          return None

    return count

  def Compile(self):
    rhyme_raw_text = self._GetBinaryOutput(self.word_queue)
    self.word_queue = []
    self._ParseRawRhymeText(rhyme_raw_text)
    # Compute syllables_per_letter
    total_letters = 0
    total_syllables = 0
    for (word, syllables) in self.num_syllables.items():
      total_letters += len(word)
      total_syllables += syllables
    self.syllables_per_letter = total_syllables / total_letters
    print("syllables_per_letter = %g" % self.syllables_per_letter)

  def _GetBinaryOutput(self, word_list):
    binary_dir = "C:\\Users\\oconaire\\Documents\\Projects\\LyricsGen\\rhyme_0.9"
    binary_args = binary_dir + "\\rhyme.exe -i"
    input_stream = "\n".join(word_list) + "\n"
    # print("INPUT:", input_stream)
    print("INCHARS:", [c for c in input_stream[:7]])
    print("Processing %d words" % len(word_list))
    # os.chdir(binary_dir)

    tmp_words_in = "words.tmp"
    with open(tmp_words_in, "wt") as fp:
      for word in word_list:
        fp.write(word+"\n")
    tmp_rhymes_out = "rhymes.tmp"
    file_in = open(tmp_words_in, "rt")
    file_out = open(tmp_rhymes_out, "wt")

    popen = subprocess.Popen(binary_args, stdin=file_in,
                             stdout=file_out, stderr=subprocess.PIPE,
                             universal_newlines=True, cwd=binary_dir)
    # popen = subprocess.Popen(binary_args, stdin=subprocess.PIPE,
    #                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    #                          universal_newlines=True, cwd=binary_dir)
    (stdoutdata, stderrdata) = popen.communicate()
    # popen.wait()

    file_in.close()
    file_out.close()

    with open(tmp_rhymes_out, "rt") as fp:
      raw_rhymes_text = fp.read()

    # os.chdir(cwd)
    # print("OUTPUT:", stdoutdata)
    # print("ERR:", stderrdata)
    return raw_rhymes_text

  """

  Parse
  Rhyming Dictionary 0.9 - interactive mode
   - exits on an empty line
   - type . to toggle between rhyme and syllable searches
   - type , to toggle between merged and non-merged results
   - type ? for help
  RHYME> Finding perfect rhymes for app...
  1: app, cap, capp, chap, chapp, clap, clapp, crap, dapp, flap, frap, gap, gapp
     hap, happ, happe, jap, kapp, klapp, knapp, krapp, lap, lapp, lappe, map, mapp
     nap, napp, pap, papp, rap, rapp, rappe, sap, sapp, schapp, schnapp, scrap
     shap, shapp, slap, snap, snapp, stapp, strap, tap, tapp, tappe, trap, trapp
     trappe, wrap, yap, yapp, zap, zapp

  2: entrap, giap, mayhap(2), recap(2), unwrap

  RHYME> Finding perfect rhymes for apple...
  2: appel, appell, apple, cappel, chapel, chappel, chappell, chapple, grapple
     happel, kappel, mapel, schappell, shappell, snapple, stapel

  RHYME> *** Word "sdvvssdfv" wasn't found

  """
  def _ParseRawRhymeText(self, rhyme_raw_text):
    def UpdateRhymes(info_queue):
      rhyming_words = [w[1] for w in info_queue]
      print("Adding %d words" % len(rhyming_words))
      for (ns, word) in info_queue:
        self.num_syllables[word] = ns
        if word not in self.rhymes:
          if " " in word:
            print("---> ", word)
            assert False
          self.rhymes[word] = rhyming_words

    info_queue = []
    lines = [line.strip() for line in rhyme_raw_text.split("\n")]
    curr_word = None
    num_syllables = None

    print("Parsing %d lines" % len(lines))
    for line in lines:
      if not line: continue
      if "Finding perfect rhymes for" in line:
        UpdateRhymes(info_queue)
        info_queue = []
        curr_word = line.split()[-1].replace("...", "")
        continue
      elif "***" in line:
        continue
      elif ":" in line:
        num_str, line = line.split(":")
        num_syllables = int(num_str)
      elif curr_word is None:
        continue
      words = line.strip().split(", ")
      for word in words:
        info_queue.append((num_syllables, word))
    if info_queue:
      UpdateRhymes(info_queue)
      info_queue = []


def LastWord(line):
  return line.split()[-1]

def MakeLimerick(poem_gen):
  # Form = AABBA
  lines = []
  poem_line = poem_gen.GetLine(9, allow_guess=True, rhyming_word=None, last_word_must_have_rhymes=True, attempts=500)
  lines.append(poem_line)
  poem_line = poem_gen.GetLine(9, allow_guess=True, rhyming_word=LastWord(poem_line), last_word_must_have_rhymes=True, attempts=500)
  lines.append(poem_line)
  poem_line = poem_gen.GetLine(5, allow_guess=True, rhyming_word=None, last_word_must_have_rhymes=True, attempts=500)
  lines.append(poem_line)
  poem_line = poem_gen.GetLine(5, allow_guess=True, rhyming_word=LastWord(poem_line), last_word_must_have_rhymes=True, attempts=500)
  lines.append(poem_line)
  count=0
  while count<200:
    poem_line = poem_gen.GetLine(9, allow_guess=True, rhyming_word=LastWord(lines[0]), last_word_must_have_rhymes=True, attempts=500)
    count += 1
    if LastWord(poem_line) != LastWord(lines[1]):
      break
  lines.append(poem_line)
  return lines






def main():
  # db_dir = "C:\\Users\\oconaire\\Documents\\Projects\\LyricsGen\\datasets\\"
  db_dir = "..\\datasets\\"
  sub_dirs_parser = {
      # "lyrics": LyricsParser,
      "books": BookParser
      }

  text_dataset = TextDataset()
  for (sub_dir, parser) in sub_dirs_parser.items():
    data_files = [f for f in os.listdir(os.path.join(db_dir, sub_dir)) if f.endswith(".txt")]
    for data_file in data_files:
      filename = os.path.join(db_dir, sub_dir, data_file)
      text_dataset.AddFile(parser(filename), source_description = filename)
  # text_dataset.AddFile(db_dir + "the_collector.txt")
  # text_dataset.AddFile(db_dir + "tool_aenima.txt")

  print("First N words:")
  print(text_dataset.GetWords()[:50])
  index = 0
  while text_dataset.GetLine(index) is not None:
    print(text_dataset.GetLine(index))
    index += 1
    if index > 49:
      print("Quitting printout at index=%d..." % index)
      break

  rhyming_dict = RhymingDictionary()
  rhyming_dict.AddWords(text_dataset.GetWords())
  rhyming_dict.Compile()

  # print(rhyming_dict.num_syllables)
  print(rhyming_dict.GetRhymes("jill"))
  print(rhyming_dict.GetRhymes("got"))

  lang_def = LanguageDefinition(text_dataset, rhyming_dict)
  for i in range(20):
    print(lang_def.GetNextWord("."))

  poem_gen = PoemGenerator(lang_def)

  for k in range(5,9):
    print("num_syllables = %d" % k)
    lines = []
    for i in range(6):
      last_word = None
      if (i % 2) == 1:
        last_word = lines[-1].split()[-1]
      poem_line = poem_gen.GetLine(k, allow_guess=True, rhyming_word=last_word, last_word_must_have_rhymes=True, attempts=500)
      lines.append(poem_line)
      print(poem_line)
      words = poem_line.split()
      for word in words:
        is_guess = lang_def.GetSyllables(word, allow_guess=False) is None
        print("  %s --> %d syllables (guess=%s)" % (word, lang_def.GetSyllables(word, allow_guess=True), str(is_guess)))
    print("POEM:%d" % k)
    for line in lines:
      print(line)

  for li in range(1,11):
    limerick = MakeLimerick(poem_gen)
    print("Limerick #%d" % li)
    for line in limerick:
      print(line)



main()

