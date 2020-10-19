from itertools import tee
from spacy.tokens import Doc
from spacy.tokens import Token

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.utils.utils import split_doc_into_sentences

Doc.set_extension('referential_cohesion_adjacent', default=[], force=True)

class ReferentialCohesionAdjacentSentencesAnalyzer:
    name = 'referential cohesion adjacent sentences analyzer'

    def __init__(self, language: str='es') -> None:
        '''
        This constructor will initialize the object that processes referential cohesion for adjacent sentences in a text. It goes after sentencizer.

        Parameters:
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self.language = language
        self.sentence_analyzer = None

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will calculate the referential cohesion between adjacent sentences in a text.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        if self.sentence_analyzer is None:
            raise AttributeError('No function to analyze referential cohesion between pair of sentences was provided.')
        # Prepare iterators to extract previous and current sentence pairs.
        sentences = split_doc_into_sentences(doc)
        
        prev, cur = tee(sentences)
        next(cur, None)
        
        doc._.referential_cohesion_adjacent = [self.sentence_analyzer(prev, cur, self.language)
                                               for prev, cur in zip(prev, cur)]
        
        return doc