import pyphen

from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Token

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES, LANGUAGES_DICTIONARY_PYPHEN
from text_complexity_analyzer_cm.utils.utils import is_word


class Syllablelizer:
    '''
    Pipe that separates the tokens in syllables. It goes after alphanumeric_word_identifier.

    The pipe adds to each alphabetic token a list of syllables as well as the count of syllables.
    '''

    name = 'Syllablelizer'

    def __init__(self, nlp: Language, language: str='es') -> None:
        '''
        This constructor will initialize the object that handles syllable processing.

        Parameters:
        nlp(Language): Spacy model used for this pipe.
        language(str): The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._nlp = nlp
        self._language = language
        self._dic = pyphen.Pyphen(lang=LANGUAGES_DICTIONARY_PYPHEN[language])
        Token.set_extension('syllables', default=[], force=True)
        Token.set_extension('syllable_count', default=0)

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find the syllables for each token that is a word.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The analyzed spacy document.
        '''
        for token in doc._.alpha_words: # Iterate every token
            token._.syllables = self._dic.inserted(token.text).split('-')
            token._.syllable_count = len(token._.syllables)
        
        return doc