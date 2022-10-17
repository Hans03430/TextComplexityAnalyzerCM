from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from typing import List


def noun_phrases_getter(doc: Doc) -> List[Span]:
    '''
    Function that returns the negative expressions as a list of Spacy Spans.

    Parameters:
    doc(Doc): A spacy doc containing the text.

    Returns:
    List[Span]: A list of spans that represent the negation expressions.
    '''
    return [doc[span['start']:span['end']]
            for span in doc._.noun_phrases_span_indices]

Doc.set_extension('noun_phrases_span_indices', force=True, default=[])
Doc.set_extension('noun_phrases', force=True, getter=noun_phrases_getter)

class NounPhraseTagger:
    '''
    This tagger has the task to find all noun phrases in a document. It needs to go after the 'Parser' pipeline component.
    '''
    name = 'noun phrase tagger'

    def __init__(self, language: str='es') -> None:
        '''
        This constructor will initialize the object that tags noun phrases.

        Parameters:
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all noun phrases and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        noun_phrases = set()
        for nc in doc.noun_chunks: # We find the noun phrases in the entire document
            for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i+1]]:
                noun_phrases.add(np)

        doc._.noun_phrases_span_indices = [{'start': span.start,
                                            'end': span.end,
                                            'label': span.label}
                                           for span in filter_spans(noun_phrases)] # Save the noun phrases found
        
        return doc