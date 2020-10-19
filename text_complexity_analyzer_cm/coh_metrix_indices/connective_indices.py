import multiprocessing
import pyphen
import spacy
import string

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs

class ConnectiveIndices:
    '''
    This class will handle all operations to obtain the connective indices of a text according to Coh-Metrix
    '''
    def __init__(self, nlp, language: str='es', descriptive_indices: DescriptiveIndices=None) -> None:
        '''
        The constructor will initialize this object that calculates the connective indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
        language(str): The language that the texts to process will have.

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
            self._di = DescriptiveIndices(language)
        else:
            self._di = descriptive_indices

    def _get_connectives_incidence(self, text: str, disable_pipeline: List, count_connectives_function: Callable, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for any connectives.

        Parameters:
        text(str): The text to be analyzed.
        disable_pipeline(List): The elements of the pipeline to be disabled.
        count_connectives_function(Callable): The function that counts any type of connectives. It takes a Spacy Doc and returns an integer.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of any connectives per {self._incidence} words.
        """
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            paragraphs = split_text_into_paragraphs(text) # Obtain paragraphs
            threads = multiprocessing.cpu_count() if workers == -1 else workers
            wc = word_count if word_count is not None else self._di.get_word_count_from_text(text)
            self._nlp.get_pipe('feature counter').counter_function = count_connectives_function
            connectives = sum(doc._.feature_count
                              for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads))

            return (connectives / wc) * self._incidence

    def get_causal_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for causal connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of causal connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['causal connective tagger', 'tagger', 'feature counter']]
        causal_connectives_counter = lambda doc: len(doc._.causal_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=causal_connectives_counter, workers=workers)

    def get_logical_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for logical connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of logical connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['logical connective tagger', 'tagger', 'feature counter']]
        logical_connectives_counter = lambda doc: len(doc._.logical_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=logical_connectives_counter, workers=workers)

    def get_adversative_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for adversative connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of adversative connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['adversative connective tagger', 'tagger', 'feature counter']]
        adversative_connectives_counter = lambda doc: len(doc._.adversative_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=adversative_connectives_counter, workers=workers)

    def get_temporal_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for temporal connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of temporal connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['temporal connective tagger', 'tagger', 'feature counter']]
        temporal_connectives_counter = lambda doc: len(doc._.temporal_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=temporal_connectives_counter, workers=workers)

    def get_additive_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for additive connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of additive connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['additive connective tagger', 'tagger', 'feature counter']]
        additive_connectives_counter = lambda doc: len(doc._.additive_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=additive_connectives_counter, workers=workers)


    def get_all_connectives_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        """
        This method returns the incidence per {self._incidence} words for all connectives.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of all connectives per {self._incidence} words.
        """
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['causal connective tagger', 'logical connective tagger', 'adversative connective tagger', 'temporal connective tagger', 'additive connective tagger', 'causal connective tagger', 'tagger', 'feature counter']]
        all_connectives_counter = lambda doc: len(doc._.causal_connectives) + len(doc._.logical_connectives) + len(doc._.adversative_connectives) + len(doc._.temporal_connectives) + len(doc._.additive_connectives)
        
        return self._get_connectives_incidence(text, disable_pipeline=disable_pipeline, count_connectives_function=all_connectives_counter, workers=workers)
