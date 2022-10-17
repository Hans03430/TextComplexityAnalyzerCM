from spacy.language import Language
from spacy.tokens import Doc

from text_complexity_analyzer_cm.utils.utils import is_word


class AlphanumericWordIdentifier:
    name = 'alphanumeric_word_identifier'

    def __init__(self, nlp: Language) -> None:
        '''
        This constructor sets the new extension attributes for Docs.

        Parameters:
        nlp(Language): The spacy model that uses this pipeline
        
        Returns:
        None.
        '''
        self._nlp = nlp

        Doc.set_extension('alpha_words', default=[])

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will identify the words that are considered alphanumeric.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        doc._.alpha_words = [token for token in doc if is_word(token)]
        
        return doc