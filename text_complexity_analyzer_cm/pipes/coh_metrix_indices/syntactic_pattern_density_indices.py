import multiprocessing

from spacy.language import Language
from spacy.tokens import Doc
from typing import Callable
from typing import List
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs


class SyntacticPatternDensityIndices:
    '''
    This class will handle all operations to find the synthactic pattern density indices of a text according to Coh-Metrix.
    '''

    name = 'syntactic_pattern_density_indices'

    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the synthactic pattern density indices for a specific language of those that are available.

        Parameters:
        nlp: The spacy model that corresponds to a language.
         
        Returns:
        None.
        '''
        required_pipes = ['negative_expression_tagger', 'noun_phrase_tagger', 'alphanumeric_word_identifier']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Syntatic pattern density indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        self._incidence = 1000
        Doc.set_extension('syntactic_pattern_density_indices', default={})

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method calculates the syntatic pattern density indices.

        Parameters:
        doc(Doc): A Spacy document.

        Reeturns:
        Doc: The spacy document that was analyzed
        '''
        doc._.syntactic_pattern_density_indices['DRNP'] = self.__get_noun_phrase_density(doc)
        doc._.syntactic_pattern_density_indices['DRVP'] = self.__get_verb_phrase_density(doc)
        doc._.syntactic_pattern_density_indices['DRNEG'] = self.__get_negation_expressions_density(doc)
        return doc

    def __get_syntactic_pattern_density(self, text: str, disable_pipeline: List, sp_counter_function: Callable=None, word_count: int=None, workers: int=-1) -> int:
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

    def __get_noun_phrase_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of noun phrases that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of noun phrases per {self._incidence} words.
        '''
        return (doc._.noun_phrases_count / doc._.alpha_words_count) * self._incidence

    def __get_verb_phrase_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of verb phrases that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of verb phrases per {self._incidence} words.
        '''
        return (doc._.verb_phrases_count / doc._.alpha_words_count) * self._incidence

    def __get_negation_expressions_density(self, doc: Doc) -> float:
        '''
        This function obtains the incidence of negation expressions that exist on a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analized.
        
        Returns:
        float: The incidence of negation expressions per {self._incidence} words.
        '''
        return (doc._.negative_expressions_count / doc._.alpha_words_count) * self._incidence