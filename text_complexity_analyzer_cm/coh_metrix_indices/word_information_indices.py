import multiprocessing
import spacy

from typing import Callable
from typing import List
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs

class WordInformationIndices:
    '''
    This class will handle all operations to obtain the word information indices of a text according to Coh-Metrix.
    '''
    def __init__(self, nlp, language: str='es', descriptive_indices: DescriptiveIndices=None) -> None:
        '''
        The constructor will initialize this object that calculates the word information indices for a specific language of those that are available.

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

    def _get_word_type_incidence(self, text: str, disable_pipeline :List, counter_function: Callable, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of a certain type of word in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        disable_pipeline(List): The pipeline elements to be disabled.
        counter_function(Callable): The function that counts the amount of a certain type of word.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of a certain type of word per {self._incidence} words.
        '''
        if len(text) == 0:
            raise ValueError('The text is empty.')
        elif workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0')
        else:
            paragraphs = split_text_into_paragraphs(text) # Obtain paragraphs
            threads = multiprocessing.cpu_count() if workers == -1 else workers
            wc = word_count if word_count is not None else self._di.get_word_count_from_text(text)
            self._nlp.get_pipe('feature counter').counter_function = counter_function
            words = sum(doc._.feature_count
                        for doc in self._nlp.pipe(paragraphs, batch_size=threads, disable=disable_pipeline, n_process=threads))

            return (words / wc) * self._incidence

    def get_noun_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of nouns in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of nouns per {self._incidence} words.
        '''
        noun_counter = lambda doc: sum(1
                                       for token in doc
                                       if is_word(token) and token.pos_ in ['NOUN', 'PROPN'])
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=noun_counter, workers=workers)

    def get_verb_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of verbs in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of verbs per {self._incidence} words.
        '''
        verb_counter = lambda doc: sum(1
                                       for token in doc
                                       if is_word(token) and token.pos_ == 'VERB')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=verb_counter, workers=workers)

    def get_adjective_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of adjectives in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of adjectives per {self._incidence} words.
        '''
        adjective_counter = lambda doc: sum(1
                                            for token in doc
                                            if is_word(token) and token.pos_ == 'ADJ')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=adjective_counter, workers=workers)

    def get_adverb_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of adverbs in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of adverbs per {self._incidence} words.
        '''
        adverb_counter = lambda doc: sum(1
                                         for token in doc
                                         if is_word(token) and token.pos_ == 'ADV')
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=adverb_counter, workers=workers)

    def get_personal_pronoun_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_first_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in first person and singular form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in first person and singular form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.tag_ and 'Person=1' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_first_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in first person and plural form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in first person and plural form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.tag_ and 'Person=1' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_second_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in second person and singular form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in second person and singular form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.tag_ and 'Person=2' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_second_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in second person and plural form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in second person and plural form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.tag_ and 'Person=2' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_third_person_singular_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in third person and singular form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in third person and singular form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Sing' in token.tag_ and 'Person=3' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)

    def get_personal_pronoun_third_person_plural_form_incidence(self, text: str, word_count: int=None, workers: int=-1) -> float:
        '''
        This method calculates the incidence of personal pronouns in third person and plural form in a text per {self._incidence} words.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words in the text.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        float: The incidence of personal pronouns in third person and plural form per {self._incidence} words.
        '''
        if self.language == 'es':
            pronoun_counter = lambda doc: sum(1
                                              for token in doc
                                              if is_word(token) and token.pos_ == 'PRON' and 'Number=Plur' in token.tag_ and 'Person=3' in token.tag_)
        
        disable_pipeline = [pipe for pipe in self._nlp.pipe_names if pipe not in ['tagger', 'feature counter']]

        return self._get_word_type_incidence(text, disable_pipeline=disable_pipeline, counter_function=pronoun_counter, workers=workers)
