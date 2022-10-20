import multiprocessing

from spacy.language import Language
from spacy.tokens import Doc
from typing import Callable
from typing import List
from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import split_text_into_paragraphs

class WordInformationIndices:
    '''
    This class will handle all operations to obtain the word information indices of a text according to Coh-Metrix.
    '''
    name = 'word_information_indices'

    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the word information indices for a specific language of those that are available.

        Parameters:
        nlp(Language): The spacy model that corresponds to a language.
        
        Returns:
        None.
        '''
        required_pipes = ['alphanumeric_word_identifier', 'informative_word_tagger']

        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Word information indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        self._incidence = 1000
        Doc.set_extension('word_information_indices', default={})

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the word information indices

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The spacy document analyzed.
        '''
        doc._.word_information_indices['WRDNOUN'] = self.__get_noun_incidence(doc)
        doc._.word_information_indices['WRDVERB'] = self.__get_verb_incidence(doc)
        doc._.word_information_indices['WRDADJ'] = self.__get_adjective_incidence(doc)
        doc._.word_information_indices['WRDADV'] = self.__get_adverb_incidence(doc)
        doc._.word_information_indices['WRDPRO'] = self.__get_personal_pronoun_incidence(doc)
        doc._.word_information_indices['WRDPRP1s'] = self.__get_personal_pronoun_first_person_singular_form_incidence(doc)
        doc._.word_information_indices['WRDPRP1p'] = self.__get_personal_pronoun_first_person_plural_form_incidence(doc)
        doc._.word_information_indices['WRDPRP2s'] = self.__get_personal_pronoun_second_person_singular_form_incidence(doc)
        doc._.word_information_indices['WRDPRP2p'] = self.__get_personal_pronoun_second_person_plural_form_incidence(doc)
        doc._.word_information_indices['WRDPRP3s'] = self.__get_personal_pronoun_third_person_singular_form_incidence(doc)
        doc._.word_information_indices['WRDPRP3p'] = self.__get_personal_pronoun_third_person_plural_form_incidence(doc)
        
        return doc

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

    def __get_noun_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of nouns in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of nouns per {self._incidence} words.
        '''
        return (doc._.nouns_count / doc._.alpha_words_count) * self._incidence

    def __get_verb_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of verb in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of verb per {self._incidence} words.
        '''
        return (doc._.verbs_count / doc._.alpha_words_count) * self._incidence

    def __get_adjective_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of adjective in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of adjective per {self._incidence} words.
        '''
        return (doc._.adjectives_count / doc._.alpha_words_count) * self._incidence

    def __get_adverb_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of adverb in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of adverb per {self._incidence} words.
        '''
        return (doc._.adverbs_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns per {self._incidence} words.
        '''
        return (doc._.pronouns_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_first_person_singular_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in first person and singular in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in first person and singular per {self._incidence} words.
        '''
        return (doc._.pronouns_singular_first_person_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_first_person_plural_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in first person and plural in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in first person and plural per {self._incidence} words.
        '''
        return (doc._.pronouns_plural_first_person_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_second_person_singular_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in second person and singular in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in second person and singular per {self._incidence} words.
        '''
        return (doc._.pronouns_singular_second_person_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_second_person_plural_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in second person and plural in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in second person and plural per {self._incidence} words.
        '''
        return (doc._.pronouns_plural_second_person_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_third_person_singular_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in third person and singular in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in third person and singular per {self._incidence} words.
        '''
        return (doc._.pronouns_singular_third_person_count / doc._.alpha_words_count) * self._incidence

    def __get_personal_pronoun_third_person_plural_form_incidence(self, doc: Doc) -> float:
        '''
        This method calculates the incidence of personal pronouns in third person and plural in a text per {self._incidence} words.

        Parameters:
        doc(Doc): The text to be analyzed.
        
        Returns:
        float: The incidence of personal pronouns in third person and plural per {self._incidence} words.
        '''
        return (doc._.pronouns_plural_third_person_count / doc._.alpha_words_count) * self._incidence
