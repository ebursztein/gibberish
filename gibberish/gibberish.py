import re
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GibberishDetector:

    def __init__(self, markov_threshold=0.05):
        self.markov_threshold = markov_threshold
        script_path = Path(__file__).resolve()
        model_file = script_path.parent / 'model' / 'probabilities.pkl'
        assert model_file.exists(), f"Model file {model_file} not found"
        with open(model_file, 'rb') as f:
            self.mdl = pickle.load(f)

        if not model_file.exists():
            logging.warning(f"Model file {model_file} not found. You may need to train the model.")
            self.mdl = None  # Set mdl to None if the model doesn't exist
        else:
            with open(model_file, 'rb') as f:
                self.mdl = pickle.load(f)

        # Inspired by nostril https://github.com/casics/nostril/
        self._simple_nonsense_re = re.compile(
            r"""
            \A[^eariotnslcu]+\Z  # Lack of common English letters
            |(.)\1{4,}           # 5+ repeated characters (adjusting for the initial match)
            |(.)\2{2,}(.)\3{2,}  # Repeating sequences (e.g., abcabcabc)
            |(.)(.)\4\5\4\5     # Repeating pairs (e.g., ababab)
            |(.)(.)(.)\6\7\8\6\7\8\6\7\8  # Repeating triplets
            """, re.I | re.VERBOSE)

        self.MIN_AVG_WORD_LENGTH = 2
        self.MAX_AVG_WORD_LENGTH = 10
        self.UNKNOWN_TRIGRAM_PROB = 1e-9
        self.MAX_WORD_LENGTH = 100


    def chargrams(self, text, n=3):
        """
        Text character n-grams (list-based).

        Args:
            text: The input text.
            n: The length of the n-grams.

        Returns:
            list: A list of n-grams.
        """
        return [text[i:i+n] for i in range(len(text)-n+1)]  # Back to list-based

    def train(self, input_dir='data/train/',
                            output_file='model/new_probabilities.pkl',
                            max_files=-1):
        input_path = Path(input_dir)
        if not input_path.exists():
            logging.error(f"Input directory {input_dir} does not exist.")
            return

        counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        total = 0
        fnames = list(input_path.glob('*.json'))
        if max_files > 0:
            fnames = fnames[:max_files]
            logging.info(f"Using {max_files} files")

        # Use Counter for character frequencies:
        char_counts = Counter()
        total_chars = 0
        word_len = [0] * self.MAX_WORD_LENGTH
        total_words = 0

        # Counters for character types:
        vowel_count_total = 0
        consonant_count_total = 0
        number_count_total = 0
        other_count_total = 0

        # Use Counter for short word frequencies:
        short_word_counts = Counter()

        pb = tqdm(total=len(fnames), desc="Counting trigrams")
        for file in fnames:
            with open(file, 'r') as f:
                data = json.load(f)
            for article in tqdm(data, desc=f"Processing {file}", leave=False):
                for g in self.chargrams(article['text'], n=3):
                    counts[g[0]][g[1]][g[2]] += 1
                    total += 1

                for c in article['text']:
                    char_counts[c] += 1  # Increment character count using Counter
                    total_chars += 1

                    # Classify and count characters:
                    if c.lower() in 'aeiouy':
                        vowel_count_total += 1
                    elif c.isalpha():
                        consonant_count_total += 1
                    elif c.isdigit():
                        number_count_total += 1
                    else:
                        other_count_total += 1

                for w in article['text'].split():
                    if len(w) < self.MAX_WORD_LENGTH:
                        word_len[len(w)] += 1
                        total_words += 1
                        if len(w) <= 3:  # Consider words of length 3 or less as "short"
                            short_word_counts[w.lower()] += 1  # Increment short word count

            pb.set_postfix({"trigrams":total, 'chars': total_chars, 'words': total_words})
            pb.update(1)
        pb.close()

        trigram_probabilities = {}
        max_proba = 0
        min_proba = float('inf')
        sum_proba = 0

        for i in tqdm(counts, desc="Computing probabilities"):
            trigram_probabilities[i] = {}
            for j, inner_counts in counts[i].items():
                trigram_probabilities[i][j] = {}
                s = sum(inner_counts.values())
                for k, count in inner_counts.items():
                    prob = math.log(count / s)
                    trigram_probabilities[i][j][k] = prob
                    max_proba = max(max_proba, prob)
                    min_proba = min(min_proba, prob)
                    sum_proba += prob

        # Normalize character frequencies using Counter.total():
        char_freq = {char: count / total_chars for char, count in char_counts.items()}

        for i in range(self.MAX_WORD_LENGTH):
            word_len[i] /= total_words

        # Get most common short words directly from Counter:
        most_popular_short_words = [word for word, count in short_word_counts.most_common(20)]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            data = {
                "mat": trigram_probabilities,
                "max": max_proba,
                "min": min_proba,
                "avg": sum_proba / total,
                "char_freq": char_freq,
                "total_chars": total_chars,
                "word_len": word_len,
                "total_words": total_words,
                "vowel_count": vowel_count_total,
                "consonant_count": consonant_count_total,
                "number_count": number_count_total,
                "other_count": other_count_total,
                "most_popular_short_words": most_popular_short_words,
            }
            logging.info(f"Saved data {data.keys()}")
            pickle.dump(data, f)
        logging.info(f"Found {total} trigrams")
        logging.info(f"Probabilities saved to {output_file}")

    def gibberish_patterns(self, text):
        """
        Checks if the text contains gibberish patterns.

        Args:
            text: The input text.

        Returns:
            True if gibberish patterns are found, False otherwise.
        """
        return bool(self._simple_nonsense_re.search(text))

    def gibberish_simple_stats(self, text):
        """
        Checks for gibberish based on simple word length statistics.

        Args:
            text: The input text.

        Returns:
            True if the text likely contains gibberish, False otherwise.
        """
        words = text.split()
        if not words:
            return True  # Empty text is considered gibberish
        wls = [len(w) for w in words]
        avg_word_len = sum(wls) / len(wls)
        return avg_word_len < self.MIN_AVG_WORD_LENGTH or avg_word_len > self.MAX_AVG_WORD_LENGTH

    def markov(self, text):
        """
        Calculates the average transition probability using a Markov model,
        incorporating repetition and length penalties.

        Args:
          text: The input text.

        Returns:
            float: The average transition probability.  Lower values indicate
            higher likelihood of gibberish.
        """
        log_prob = 0.0
        transition_ct = 0
        repetition_penalty_exponent = 1.5  # Tune this value
        length_penalty_threshold = 25      # Tune this value
        length_penalty_exponent = 1.2     # Tune this value


        if self.mdl is None:
            logging.warning("Markov model not loaded.  Returning 0.0 probability.")
            return 0.0  # Or some other default value

        mdl = self.mdl
        last_trigram = None  # Keep track of the previous trigram
        repetition_count = 0

        for a, b, c in self.chargrams(text, 3):
            trigram = (a, b, c)

            # Add-k (Laplace) smoothing:
            k = 0.01  # Smoothing parameter.  Tune this!  Smaller k = less smoothing.
            unknown_probability = math.log(k / (self.mdl['total_chars'] + k * (len(self.mdl['char_freq'])**3)))

            prob = mdl['mat'].get(a, {}).get(b, {}).get(c, unknown_probability)

            # Repetition penalty:
            if trigram == last_trigram:
                repetition_count += 1
            else:
                repetition_count = 0  # Reset if the trigram changes
            last_trigram = trigram

            # Apply repetition penalty to the log_prob:
            prob -= repetition_count * repetition_penalty_exponent

            log_prob += prob
            transition_ct += 1

        # FIXME: Length penalty is disabled for now as it needs tuning
        # Length penalty (applied to the final log_prob):
        # length_penalty = pow(max(0, transition_ct - length_penalty_threshold), length_penalty_exponent)
        # log_prob -= length_penalty  # Subtract the penalty

        return math.exp(log_prob / (transition_ct or 1))


    # fast check for gibberish
    def check_vowel_consonant_ratio(self, text):
        if self.mdl is None:
            return False  # No model

        vowels = "aeiouyAEIOUY"
        vowel_count = sum(1 for char in text if char in vowels)
        consonant_count = sum(1 for char in text if char.isalpha() and char not in vowels)

        if consonant_count == 0:
            return True

        ratio = vowel_count / consonant_count

        # Calculate expected ratio from training data:
        expected_ratio = self.mdl['vowel_count'] / self.mdl['consonant_count']
        # Use a threshold relative to the expected ratio:
        tolerance = 0.5
        upper_bound = expected_ratio * (1 + tolerance)
        lower_bound = expected_ratio * (1 - tolerance)
        print(f"Vowel/consonant ratio: {ratio:.4f} (expected: {expected_ratio:.4f}) (bounds: {lower_bound:.4f} - {upper_bound:.4f})")

        return ratio < expected_ratio * (1 - tolerance) or ratio > expected_ratio * (1 + tolerance)


    def check_character_frequency(self, text):
        if self.mdl is None:
            return False # No model, can't do the check.

        expected_freqs = self.mdl['char_freq']
        text_freqs = Counter()

        for char in text:
            text_freqs[char.lower()] += 1  # Case-insensitive
        total_chars = len(text)
        if total_chars ==0:
            return True

        for char in text_freqs:
            text_freqs[char] /= total_chars

        deviation = 0
        for char, expected_freq in expected_freqs.items():
            deviation += abs(text_freqs.get(char, 0) - expected_freq) #get is needed if a char isn't found

        # Adjust threshold based on your data
        threshold = 0.5
        return deviation > threshold

    def check_short_words(self, text):
        if self.mdl is None:
            return False  # No model loaded

        words = text.split()
        for word in words:
            if word.lower() in self.mdl['most_popular_short_words']:
                return False  # Found a common short word

        return True  # No common short words -> potentially gibberish

    def check_repeated_words(self, text):
        words = text.split()
        repetition_threshold = 0.3  # same words is X% of the text
        if len(words) < 3:
            return False  # Too short to check for repetition

        # count and take the max
        cnts = Counter(words)
        most_common, most_common_count = cnts.most_common(1)[0]
        if most_common_count / len(words) >= repetition_threshold:
            return True

        return False


    def is_text(self, snippet: str, debug: bool = False) -> float:
        """
        Determines if a snippet is real text based on several checks.

        Args:
            snippet: The input text snippet.

        Returns:
            1.0 if the snippet is likely real text, 0.0 if it's likely gibberish.
        """
        if self.gibberish_patterns(snippet):
            if debug:
                return 0.0, "Gibberish patterns"
            return 0.0

        if self.gibberish_simple_stats(snippet):
            if debug:
                return 0.0, "Simple stats"
            return 0.0

        # experimental fast check
        # if self.check_vowel_consonant_ratio(snippet):
        #     if debug:
        #         return 0.0, "Vowel/consonant ratio"
        #     return 0.0

        # if self.check_short_words(snippet):
        #     if debug:
        #         return 0.0, "Short words"
        #     return 0.0

        if self.check_repeated_words(snippet):
            if debug:
                return 0.0, "Repeated words"
            return 0.0

        # if self.check_character_frequency(snippet):
        #     if debug:
        #         return 0.0, "Character frequency"
        #     return 0.0


        markov_proba = self.markov(snippet)
        if markov_proba <= self.markov_threshold:
            if debug:
                return 0.0, f"Markov probability {markov_proba:.4f} below threshold {self.markov_threshold}"
            return 0.0

        return 1.0, f"OK markov probability {markov_proba:.4f}"