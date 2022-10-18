import multiprocessing

import spacy

from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES


class ReadabilityIndices:
    '''
    This class will handle all operations to find the readability indices of a text according to Coh-Metrix.
    '''

    def __init__(self, nlp, language: str='es', descriptive_indices: DescriptiveIndices=None) -> None:
        '''
        The constructor will initialize this object that calculates the readability indices for a specific language of those that are available.

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

        if descriptive_indices is None: # Assign the descriptive indices to an attribute
            self._di = DescriptiveIndices(language=language, nlp=nlp)
        else:
            self._di = descriptive_indices

    def calculate_fernandez_huertas_grade_level(self, text: str=None, mean_syllables_per_word: int=None, mean_words_per_sentence: int=None, workers: int=-1) -> float:
        '''
        This function obtains the Fernández-Huertas readability index for a text.

        Parameters:
        text(str): The text to be analized.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        mean_syllables_per_word(int): The mean of syllables per word in the text.
        mean_words_per_sentence(int): The mean amount of words per sentences in the text.

        Returns:
        float: The Fernández-Huertas readability index for a text.
        '''
        if self.language != 'es':
            raise ValueError('This readability index is for spanish.')
        elif text is not None and len(text) == 0:
            raise ValueError('The word is empty.')
        elif text is None and (mean_syllables_per_word is None or mean_words_per_sentence is None):
            raise ValueError('If there\'s no text, then you must pass mean_syllables_per_word and mean_words_per_sentence at the same time.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            threads = multiprocessing.cpu_count() if workers == -1 else workers
            mspw = mean_syllables_per_word if mean_syllables_per_word is not None else self._di.get_mean_of_syllables_per_word(text=text, workers=threads)
            mwps = mean_words_per_sentence if mean_words_per_sentence is not None else self._di.get_mean_of_length_of_sentences(text=text, workers=threads)
            
            return 206.84 - 0.6 * mspw - 1.02 * mwps