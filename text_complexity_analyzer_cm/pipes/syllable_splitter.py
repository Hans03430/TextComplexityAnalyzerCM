import pyphen

from spacy.tokens import Doc
from spacy.tokens import Token

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES, LANGUAGES_DICTIONARY_PYPHEN
from text_complexity_analyzer_cm.utils.utils import is_word

Token.set_extension('syllables', default=None, force=True)

class SyllableSplitter:
    name = 'syllable splitter'

    def __init__(self, language: str='es') -> None:
        '''
        This constructor will initialize the object that handles syllable processing.

        Parameters:
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._dic = pyphen.Pyphen(lang=LANGUAGES_DICTIONARY_PYPHEN[language])

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find the syllables for each token that is a word.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        for token in doc: # Iterate every token
            if is_word(token):
                token._.syllables = self._dic.inserted(token.text).split('-')
        
        return doc