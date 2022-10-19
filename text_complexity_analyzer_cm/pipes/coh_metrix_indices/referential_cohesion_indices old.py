import multiprocessing
import spacy
import statistics

from itertools import combinations
from spacy.tokens import Span
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.statistics_results import StatisticsResults
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import is_content_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs
from typing import Callable
from typing import List


class ReferentialCohesionIndices:
    '''
    This class will handle all operations to find the synthactic pattern density indices of a text according to Coh-Metrix.
    '''
    # TODO: Implement multiprocessing
    def __init__(self, nlp, language: str='es') -> None:
        '''
        The constructor will initialize this object that calculates the synthactic pattern density indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
        language(str): The language that the texts to process will have.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self.language = language
        self._nlp = nlp

    def _calculate_overlap_for_adjacent_sentences(self, text: str, disable_pipeline: List, sentence_analyzer: Callable, statistic_type: str='mean', workers: int=-1) -> StatisticsResults:
        '''
        This method calculates the overlap for adjacent sentences in a text. MULTIPROCESSING STILL NOT IMPLEMENTED.

        Parameters:
        text(str): The text to be analyzed.
        disable_pipeline(List): The pipeline elements to be disabled.
        sentence_analyzer(Callable): The function that analyzes sentences to check cohesion.
        statistic_type(str): Whether to calculate the mean and/or the standard deviation. It accepts 'mean', 'std' or 'all'.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        StatisticsResults: The standard deviation and mean of the overlap.
        '''
        # TODO MULTIPROCESSING. WORKERS IS JUST A PLACEHOLDER
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif statistic_type not in ['mean', 'std', 'all']:
            raise ValueError('\'statistic_type\' can only take \'mean\', \'std\' or \'all\'.')
        else:
            self._nlp.get_pipe('referential cohesion adjacent sentences analyzer').sentence_analyzer = sentence_analyzer
            doc = self._nlp(text, disable=disable_pipeline)
            stat_results = StatisticsResults() # Create empty container

            if len(doc._.referential_cohesion_adjacent) == 0:
                return stat_results
            else:
                if statistic_type in ['mean', 'all']:
                    stat_results.mean = statistics.mean(doc._.referential_cohesion_adjacent)

                if statistic_type in ['std', 'all']:
                    stat_results.std = statistics.pstdev(doc._.referential_cohesion_adjacent)
                
                return stat_results

    def _calculate_overlap_for_all_sentences(self, text: str, disable_pipeline: List, sentence_analyzer: Callable, statistic_type: str='all', workers: int=-1) -> StatisticsResults:
        '''
        This method calculates the overlap for all sentences in a text. MULTIPROCESSING STILL NOT IMPLEMENTED.

        Parameters:
        text(str): The text to be analyzed.
        disable_pipeline(List): The pipeline elements to be disabled.
        sentence_analyzer(Callable): The function that analyzes sentences to check cohesion.
        statistic_type(str): Whether to calculate the mean and/or the standard deviation. It accepts 'mean', 'std' or 'all'.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        StatisticsResults: The standard deviation and mean of the overlap.
        '''
        # TODO MULTIPROCESSING. WORKERS IS JUST A PLACEHOLDER.
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif statistic_type not in ['mean', 'std', 'all']:
            raise ValueError('\'statistic_type\' can only take \'mean\', \'std\' or \'all\'.')
        else:
            self._nlp.get_pipe('referential cohesion all sentences analyzer').sentence_analyzer = sentence_analyzer
            doc = self._nlp(text, disable=disable_pipeline)
            stat_results = StatisticsResults() # Create empty container

            if len(doc._.referential_cohesion_all) == 0:
                return stat_results
            else:
                if statistic_type in ['mean', 'all']:
                    stat_results.mean = statistics.mean(doc._.referential_cohesion_all)

                if statistic_type in ['std', 'all']:
                    stat_results.std = statistics.pstdev(doc._.referential_cohesion_all)
                
                return stat_results

    def get_noun_overlap_adjacent_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the noun overlap for adjacent sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean noun overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion adjacent sentences analyzer']]
        return self._calculate_overlap_for_adjacent_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_noun_overlap, statistic_type='mean').mean

    def get_noun_overlap_all_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the noun overlap for all sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean noun overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion all sentences analyzer']]
        return self._calculate_overlap_for_all_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_noun_overlap, statistic_type='mean').mean

    def get_argument_overlap_adjacent_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the argument overlap for adjacent sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean argument overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion adjacent sentences analyzer']]
        return self._calculate_overlap_for_adjacent_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_argument_overlap, statistic_type='mean').mean

    def get_argument_overlap_all_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the argument overlap for all sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean argument overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion all sentences analyzer']]
        return self._calculate_overlap_for_all_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_argument_overlap, statistic_type='mean').mean

    def get_stem_overlap_adjacent_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the stem overlap for adjacent sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean stem overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion adjacent sentences analyzer']]
        return self._calculate_overlap_for_adjacent_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_stem_overlap, statistic_type='mean').mean

    def get_stem_overlap_all_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the stem overlap for all sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean stem overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion all sentences analyzer']]
        return self._calculate_overlap_for_all_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_stem_overlap, statistic_type='mean').mean

    def get_content_word_overlap_adjacent_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the mean and standard deviation of the content word overlap for adjacent sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean mean and standard deviation of the content word overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion adjacent sentences analyzer']]
        return self._calculate_overlap_for_adjacent_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_content_word_overlap, statistic_type='all')

    def get_content_word_overlap_all_sentences(self, text: str, workers: int=-1) -> StatisticsResults:
        '''
        This method calculates the mean and standard deviation of the content word overlap for all sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        StatisticsResults: The mean mean and standard deviation of the content word overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion all sentences analyzer']]
        return self._calculate_overlap_for_all_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_content_word_overlap, statistic_type='all')

    def get_anaphore_overlap_adjacent_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the mean of the anaphore overlap for adjacent sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean mean of the anaphore overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion adjacent sentences analyzer']]
        return self._calculate_overlap_for_adjacent_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_anaphore_overlap, statistic_type='all').mean

    def get_anaphore_overlap_all_sentences(self, text: str, workers: int=-1) -> float:
        '''
        This method calculates the mean of the anaphore overlap for all sentences in a text.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The mean mean of the anaphore overlap.
        '''
        disable_pipeline = [pipe
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['sentencizer', 'tagger', 'referential cohesion all sentences analyzer']]
        return self._calculate_overlap_for_all_sentences(text=text, workers=workers, disable_pipeline=disable_pipeline, sentence_analyzer=analyze_anaphore_overlap, statistic_type='all').mean

def analyze_noun_overlap(prev_sentence: Span, cur_sentence: Span, language: str='es') -> int:
    '''
    This function analyzes whether or not there's noun overlap between two sentences for a language.

    Parameters:
    prev_sentence(Span): The previous sentence to analyze.
    cur_sentence(Span): The current sentence to analyze.
    language(str): The language of the sentences.

    Returns:
    int: 1 if there's overlap between the two sentences and 0 if no.
    '''
    # Place the tokens in a dictionary for search efficiency
    prev_sentence_noun_tokens = {token.text.lower(): None
                                 for token in prev_sentence
                                 if is_word(token) and token.pos_ == 'NOUN'}

    for token in cur_sentence:
        if language == 'es':
            if is_word(token) and token.pos_ == 'NOUN' and token.text.lower() in prev_sentence_noun_tokens:
                return 1 # There's cohesion

    return 0 # No cohesion


def analyze_argument_overlap(prev_sentence: Span, cur_sentence: Span, language: str='es') -> int:
    '''
    This function analyzes whether or not there's argument overlap between two sentences.

    Parameters:
    prev_sentence(Span): The previous sentence to analyze.
    cur_sentence(Span): The current sentence to analyze.
    language(str): The language of the sentences.

    Returns:
    int: 1 if there's overlap between the two sentences and 0 if no.
    '''
    # Place the tokens in a dictionary for search efficiency
    prev_sentence_noun_tokens = {token.lemma_.lower(): None
                                 for token in prev_sentence
                                 if is_word(token) and token.pos_ == 'NOUN'}

    prev_sentence_personal_pronouns_tokens = {token.text.lower(): None
                                              for token in prev_sentence
                                              if is_word(token) and 'PronType=Prs' in token.tag_}

    for token in cur_sentence: # Iterate every token of the current sentence
        if language == 'es':
            if is_word(token) and token.pos_ == 'NOUN' and token.lemma_.lower() in prev_sentence_noun_tokens:
                return 1 # There's cohesion by noun lemma

            if is_word(token) and 'PronType=Prs' in token.tag_ and token.text.lower() in prev_sentence_personal_pronouns_tokens:
                return 1 # There's cohesion by personal pronoun

    return 0 # No cohesion


def analyze_stem_overlap(prev_sentence: Span, cur_sentence: Span, language: str='es') -> int:
    '''
    This function analyzes whether or not there's stem overlap between two sentences.

    Parameters:
    prev_sentence(Span): The previous sentence to analyze.
    cur_sentence(Span): The current sentence to analyze.
    language(str): The language of the sentences.

    Returns:
    int: 1 if there's overlap between the two sentences and 0 if no.
    '''
    # Place the tokens in a dictionary for search efficiency
    prev_sentence_content_stem_tokens = {token.lemma_.lower(): None
                                         for token in prev_sentence
                                         if is_content_word(token)}

    for token in cur_sentence:
        if language == 'es':
            if is_word(token) and token.pos_ in ['NOUN', 'PROPN'] and token.lemma_.lower() in prev_sentence_content_stem_tokens:
                return 1 # There's cohesion

    return 0 # No cohesion


def analyze_content_word_overlap(prev_sentence: Span, cur_sentence: Span, language='es') -> float:
    '''
    This function calculates the proportional content word overlap between two sentences.

    Parameters:
    prev_sentence(Span): The previous sentence to analyze.
    cur_sentence(Span): The current sentence to analyze.
    language(str): The language of the sentences.

    Returns:
    float: Proportion of tokens that overlap between the current and previous sentences
    '''
    total_tokens = len([token for token in prev_sentence if is_content_word(token)]) + len([token for token in cur_sentence if is_content_word(token)])

    if total_tokens == 0: # Nothing to compute
        return 0
    else:
        prev_sentence_content_words_tokens = {token.text.lower(): None
                                              for token in prev_sentence
                                              if is_content_word(token)}
        matches = 0 # Matcher counter

        for token in cur_sentence:
            if language == 'es':
                if is_content_word(token) and token.text.lower() in prev_sentence_content_words_tokens:
                    matches += 2 # There's cohesion

        return matches / total_tokens


def analyze_anaphore_overlap(prev_sentence: Span, cur_sentence: Span, language: str='es') -> int:
    '''
    This function analyzes whether or not there's anaphore overlap between two sentences.

    Parameters:
    prev_sentence(Span): The previous sentence to analyze.
    cur_sentence(Span): The current sentence to analyze.
    language(str): The language of the sentences.

    Returns:
    int: 1 if there's overlap between the two sentences and 0 if no.
    '''
    # Place the tokens in a dictionary for search efficiency
    prev_sentence_pronoun_tokens = {token.text.lower(): None
                                    for token in prev_sentence
                                    if is_word(token) and token.pos_ == 'PRON'}

    for token in cur_sentence:
        if language == 'es':
            if is_word(token) and token.pos_ == 'PRON' and token.text.lower() in prev_sentence_pronoun_tokens:
                return 1 # There's cohesion

    return 0 # No cohesion
