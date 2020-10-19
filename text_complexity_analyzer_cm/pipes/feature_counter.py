from itertools import tee
from spacy.tokens import Doc
from spacy.tokens import Token

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

Doc.set_extension('feature_count', default=None, force=True)

class FeatureCounter:
    name = 'feature counter'

    def __init__(self, language: str='es') -> None:
        '''
        This constructor will initialize the object that counts a certain feature specified by a function. It is the last pipeline component.

        Parameters:
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self.language = language
        self.counter_function = None

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the the 'counter_function' on a text. Said function will be handle different counting.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        if self.counter_function is None:
            raise AttributeError('No function to count features was provided.')
        # Prepare iterators to extract previous and current sentence pairs.
        doc._.feature_count = self.counter_function(doc)

        return doc