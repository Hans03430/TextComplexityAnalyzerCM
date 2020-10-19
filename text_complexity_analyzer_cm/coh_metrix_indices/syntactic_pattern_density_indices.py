import multiprocessing

import spacy

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs


class SyntacticPatternDensityIndices:
    '''
    This class will handle all operations to find the synthactic pattern density indices of a text according to Coh-Metrix.
    '''

    def __init__(self, nlp, language: str='es', descriptive_indices: DescriptiveIndices=None) -> None:
        '''
        The constructor will initialize this object that calculates the synthactic pattern density indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
        language(str): The language that the texts to process will have.
        descriptive_indices(DescriptiveIndices): The class that calculates the descriptive indices of a text in a certain language.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')
        elif descriptive_indices is not None and descriptive_indices.language != language:
            raise ValueError(f'The descriptive indices analyzer must be of the same language as the word information analyzer.')
        
        self.language = language
        self._nlp = nlp
        self._incidence = 1000

        if descriptive_indices is None: # Assign the descriptive indices to an attribute
            self._di = DescriptiveIndices(language=language, nlp=nlp)
        else:
            self._di = descriptive_indices

    def _get_syntactic_pattern_density(self, text: str, disable_pipeline: List, sp_counter_function: Callable=None, word_count: int=None, workers: int=-1) -> int:
        '''
        This function obtains the incidence of a syntactic pattern that exist on a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analized.
        disable_pipeline(List): The pipeline elements to be disabled.
        sp_counter_function(Callable): The function that counts a syntactic pattern for a Spacy document. It returns an integer.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        int: The incidence of a syntactic pattern per {self._incidence} words.
        '''
        if len(text) == 0:
            raise ValueError('The word is empty.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            paragraphs = split_text_into_paragraphs(text) # Find all paragraphs
            threads = multiprocessing.cpu_count() if workers == -1 else workers
            wc = word_count if word_count is not None else self._di.get_word_count_from_text(text)            
            self._nlp.get_pipe('feature counter').counter_function = sp_counter_function
            density = sum(doc._.feature_count
                          for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads)) # Calculate with multiprocessing 
            
            return (density / wc) * self._incidence

    def get_noun_phrase_density(self, text: str, word_count: int=None, workers: int=-1) -> int:
        '''
        This function obtains the incidence of noun phrases that exist on a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analized.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        int: The incidence of noun phrases per {self._incidence} words.
        '''
        count_noun_phrases = lambda doc: len(doc._.noun_phrases)
        disable_pipeline = [pipe 
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['noun phrase tagger', 'tagger', 'parser', 'feature counter']]

        return self._get_syntactic_pattern_density(text, disable_pipeline=disable_pipeline, sp_counter_function=count_noun_phrases, workers=workers)

    def get_verb_phrase_density(self, text: str, word_count: int=None, workers: int=-1) -> int:
        '''
        This function obtains the incidence of verb phrases that exist on a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analized.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        int: The incidence of verb phrases per {self._incidence} words.
        '''
        count_verb_phrases = lambda doc: len(doc._.verb_phrases)
        disable_pipeline = [pipe 
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['verb phrase tagger', 'tagger', 'feature counter']]

        return self._get_syntactic_pattern_density(text, disable_pipeline=disable_pipeline, sp_counter_function=count_verb_phrases, workers=workers)
            
    def get_negation_expressions_density(self, text: str, word_count: int=None, workers: int=-1) -> int:
        '''
        This function obtains the incidence of negation expressions that exist on a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analized.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        int: The incidence of negation expressions per {self._incidence} words.
        '''
        count_negation_expressions = lambda doc: len(doc._.negation_expressions)
        disable_pipeline = [pipe 
                            for pipe in self._nlp.pipe_names
                            if pipe not in ['negative expression tagger', 'tagger', 'feature counter']]

        return self._get_syntactic_pattern_density(text, disable_pipeline=disable_pipeline, sp_counter_function=count_negation_expressions, workers=workers)
