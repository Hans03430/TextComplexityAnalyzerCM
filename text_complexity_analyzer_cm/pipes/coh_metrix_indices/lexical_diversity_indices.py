from spacy.language import Language
from spacy.tokens import Doc
from time import time


class LexicalDiversityIndices:
    '''
    This class will handle all operations to obtain the lexical diversity indices of a text according to Coh-Metrix
    '''
    name = 'lexical_diversity_indices'

    def __init__(self, nlp: Language) -> None:
        '''
        The constructor will initialize this object that calculates the lexical diversity indices for a specific language of those that are available. It needs the following pipes to be added before it: Content word identifier and alphanumeric word identifier.

        Parameters:
        nlp: The spacy model that corresponds to a language.

        Returns:
        None.
        '''
        required_pipes = ['content_word_identifier', 'alphanumeric_word_identifier']
        if not all((
            pipe in nlp.pipe_names
            for pipe in required_pipes
        )):
            message = 'Lexical diversity indices pipe need the following pipes: ' + ', '.join(required_pipes)
            raise AttributeError(message)
        
        self._nlp = nlp
        Doc.set_extension('lexical_diversity_indices', default=dict()) # Dictionary

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the lexical diversity indices.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The processed doc.
        '''
        if len(doc.text) == 0:
            raise ValueError('The text is empty.')

        print('Analyzing lexical diversity indices')
        start = time()
        doc._.lexical_diversity_indices['LDTTRa'] = self.__get_type_token_ratio_between_all_words(doc)
        doc._.lexical_diversity_indices['LDTTRcw'] = self.__get_type_token_ratio_of_content_words(doc)
        end = time()
        print(f'Lexical diversity indices analyzed in {end - start} seconds.')
        return doc

    def __get_type_token_ratio_between_all_words(self, doc: Doc) -> float:
        """
        This method returns the type token ratio between all words of a text.

        Parameters:
        doc(Doc): The text to be anaylized.
        
        Returns:
        float: The type token ratio between all words of a text.
        """
        return 0 if doc._.alpha_words_count == 0 else doc._.alpha_words_different_count / doc._.alpha_words_count

    def __get_type_token_ratio_of_content_words(self, doc: Doc) -> float:
        """
        This method returns the type token ratio of content words of a text. Content words are nouns, verbs, adjectives and adverbs.

        Parameters:
        doc(Doc): The text to be anaylized.
        
        Returns:
        float: The type token ratio between the content words of a text.
        """
        return 0 if doc._.content_words_count == 0 else doc._.content_words_different_count / doc._.content_words_count
