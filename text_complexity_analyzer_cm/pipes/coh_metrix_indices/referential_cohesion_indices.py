import multiprocessing
import spacy
import statistics

from itertools import combinations
from itertools import tee
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.statistics_results import StatisticsResults
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import is_content_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs
from typing import Callable, Iterator
from typing import Tuple
from typing import List


def doc_adjacent_sentence_pairs_getter(doc: Doc) -> Tuple[Span, Span]:
    '''
    Iterator that returns all pairs of adjacent sentences.
    
    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    Tuple[Span, Span]: Pair of spans that represent two adjacent sentences.
    '''
    sentences = doc._.non_empty_sentences
    prev, cur = tee(sentences)
    next(cur, None)
    # Return each pair of sentences
    for prev, cur in zip(prev, cur):
        yield prev, cur

def doc_all_sentence_pairs_getter(doc: Doc) -> Tuple[Span, Span]:
    '''
    Iterator that returns all pairs of sentences.
    
    Parameters:
    doc(Doc): The document to analyze.

    Yields:
    Tuple[Span, Span]: Pair of spans that represent a pair of sentences.
    '''
    sentences = doc._.non_empty_sentences
    # Return each pair of sentences
    for prev, cur in combinations(sentences, 2):
        yield prev, cur

class ReferentialCohesionIndices:
    '''
    This class will handle all operations to find the synthactic pattern density indices of a text according to Coh-Metrix.
    '''

    name = 'referential_cohesion_indices'

    # TODO: Implement multiprocessing
    def __init__(
        self,
        nlp: Language,
        noun_overlap_func: Callable,
        argument_overlap_func: Callable,
        stem_overlap_func: Callable,
        content_word_overlap_func: Callable,
        anaphore_overlap_func: Callable
    ) -> None:
        '''
        The constructor will initialize this object that calculates the synthactic pattern density indices for a specific language of those that are available.

        Parameters:
        nlp(Language): The spacy model that corresponds to a language.

        Returns:
        None.
        '''
        required_pipes = ['cohesion_words_tokenizer']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Referential cohesion indices need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)

        self._nlp = nlp
        self._noun_overlap_func = noun_overlap_func
        self._argument_overlap_func = argument_overlap_func
        self._stem_overlap_func = stem_overlap_func
        self._content_word_overlap_func = content_word_overlap_func
        self._anaphore_overlap_func = anaphore_overlap_func
        Doc.set_extension('referential_cohesion_indices', default={})
        Doc.set_extension('adjacent_sentence_pairs', getter=doc_adjacent_sentence_pairs_getter)
        Doc.set_extension('all_sentence_pairs', getter=doc_all_sentence_pairs_getter)

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the referential cohesion indices.

        Parameters:
        doc(Doc): A Spacy document.
        
        Returns:
        Doc: The analyzed spacy document.
        '''
        if len(doc.text) == 0:
            raise ValueError('The text is empty.')

        doc._.referential_cohesion_indices['CRFNO1'] = self.__get_noun_overlap_adjacent_sentences(doc)
        doc._.referential_cohesion_indices['CRFNOa'] = self.__get_noun_overlap_all_sentences(doc)
        doc._.referential_cohesion_indices['CRFAO1'] = self.__get_argument_overlap_adjacent_sentences(doc)
        doc._.referential_cohesion_indices['CRFAOa'] = self.__get_argument_overlap_all_sentences(doc)
        doc._.referential_cohesion_indices['CRFSO1'] = self.__get_stem_overlap_adjacent_sentences(doc)
        doc._.referential_cohesion_indices['CRFSOa'] = self.__get_stem_overlap_all_sentences(doc)
        self.__get_content_word_overlap_adjacent_sentences(doc)
        self.__get_content_word_overlap_all_sentences(doc)
        doc._.referential_cohesion_indices['CRFANP1'] = self.__get_anaphore_overlap_adjacent_sentences(doc)
        doc._.referential_cohesion_indices['CRFANPa'] = self.__get_anaphore_overlap_all_sentences(doc)
        
        return doc

    def __calculate_overlap_for_sentences(self, doc: Doc, sentence_analyzer: Callable, sentence_pairs: Iterator,statistic_type: str='mean') -> StatisticsResults:
        '''
        This method calculates the overlap for adjacent sentences in a text. MULTIPROCESSING STILL NOT IMPLEMENTED.

        Parameters:
        doc(Doc): The text to be analyzed.
        sentence_analyzer(Callable): The function that analyzes sentences to check cohesion.
        sentence_pairs(Iterator): The iterator that returns the pair of sentences
        statistic_type(str): Whether to calculate the mean and/or the standard deviation. It accepts 'mean', 'std' or 'all'.
        
        Returns:
        StatisticsResults: The standard deviation and mean of the overlap.
        '''
        # TODO MULTIPROCESSING. WORKERS IS JUST A PLACEHOLDER
        if statistic_type not in ['mean', 'std', 'all']:
            raise ValueError('\'statistic_type\' can only take \'mean\', \'std\' or \'all\'.')
        else:
            referential_cohesion = [sentence_analyzer(prev, cur) for prev, cur in sentence_pairs]
            stat_results = StatisticsResults() # Create empty container

            if len(referential_cohesion) == 0:
                return stat_results
            else:
                if statistic_type in ['mean', 'all']:
                    stat_results.mean = statistics.mean(referential_cohesion)

                if statistic_type in ['std', 'all']:
                    stat_results.std = statistics.pstdev(referential_cohesion)
                
                return stat_results
    
    def __get_noun_overlap_adjacent_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the noun overlap for adjacent sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        float: The mean noun overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._noun_overlap_func, sentence_pairs=doc._.adjacent_sentence_pairs, statistic_type='mean').mean

    def __get_noun_overlap_all_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the noun overlap for all sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        float: The mean noun overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._noun_overlap_func, sentence_pairs=doc._.all_sentence_pairs, statistic_type='mean').mean

    def __get_argument_overlap_adjacent_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the argument overlap for adjacent sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The mean argument overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._argument_overlap_func, sentence_pairs=doc._.adjacent_sentence_pairs, statistic_type='mean').mean

    def __get_argument_overlap_all_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the argument overlap for all sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        float: The mean argument overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._argument_overlap_func, sentence_pairs=doc._.all_sentence_pairs, statistic_type='mean').mean

    def __get_stem_overlap_adjacent_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the argument overlap for stem sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The mean stem overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._stem_overlap_func, sentence_pairs=doc._.adjacent_sentence_pairs, statistic_type='mean').mean

    def __get_stem_overlap_all_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the stem overlap for all sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        float: The mean stem overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._stem_overlap_func, sentence_pairs=doc._.all_sentence_pairs, statistic_type='mean').mean

    def __get_content_word_overlap_adjacent_sentences(self, doc: Doc) -> None:
        '''
        This method calculates the mean and standard deviation of the content word overlap for adjacent sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        None:
        '''
        results = self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._content_word_overlap_func, sentence_pairs=doc._.adjacent_sentence_pairs, statistic_type='all')
        doc._.referential_cohesion_indices['CRFCWO1'] = results.mean
        doc._.referential_cohesion_indices['CRFCWO1d'] = results.std        

    def __get_content_word_overlap_all_sentences(self, doc: Doc) -> StatisticsResults:
        '''
        This method calculates the mean and standard deviation of the content word overlap for all sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        None:
        '''
        results = self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._content_word_overlap_func, sentence_pairs=doc._.all_sentence_pairs, statistic_type='all')
        doc._.referential_cohesion_indices['CRFCWOa'] = results.mean
        doc._.referential_cohesion_indices['CRFCWOad'] = results.std 

    def __get_anaphore_overlap_adjacent_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the argument overlap for anaphore sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The mean anaphore overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._anaphore_overlap_func, sentence_pairs=doc._.adjacent_sentence_pairs, statistic_type='mean').mean

    def __get_anaphore_overlap_all_sentences(self, doc: Doc) -> float:
        '''
        This method calculates the anaphore overlap for all sentences in a text.

        Parameters:
        doc(Doc): The text to be analyzed.

        Returns:
        float: The mean anaphore overlap.
        '''
        return self.__calculate_overlap_for_sentences(doc=Doc, sentence_analyzer=self._anaphore_overlap_func, sentence_pairs=doc._.all_sentence_pairs, statistic_type='mean').mean
