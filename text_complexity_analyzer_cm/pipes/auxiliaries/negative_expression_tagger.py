from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from typing import List

def negation_expressions_getter(doc: Doc) -> List[Span]:
    '''
    Function that returns the negative expressions as a list of Spacy Spans.

    Parameters:
    doc(Doc): A spacy doc containing the text.

    Returns:
    List[Span]: A list of spans that represent the negation expressions.
    '''
    return [doc[span['start']:span['end']]
            for span in doc._.negation_expressions_span_indices]

Doc.set_extension('negation_expressions_span_indices', default=[], force=True)
Doc.set_extension('negation_expressions', force=True, getter=negation_expressions_getter)

class NegativeExpressionTagger:
    '''
    This tagger has the task to find all verb phrases in a document. It needs to go after the 'Parser' pipeline component.
    '''
    name = 'negative expression tagger'

    def __init__(self, nlp, language: str='es') -> None:
        '''
        This constructor will initialize the object that tags verb phrases.

        Parameters:
        nlp: The Spacy model to use this tagger with.
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._matcher = Matcher(nlp.vocab)

        if language == 'es': # Negative expressions for spanish
            self._pattern = [{'POS': 'ADV', 
                            'LOWER': {
                                'IN': ['no', 'nunca', 'jamás', 'tampoco']
                                }
                            }] # The pattern for negation expressions
        else: # Support for future languages
            pass

        self._matcher.add('negation expression', None, self._pattern) # Add the negative expression pattern

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all negation expressions and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        matches = self._matcher(doc)
        negation_expression_spans = [doc[start:end] for _, start, end in matches]

        doc._.negation_expressions_span_indices = [{'start': span.start,
                                                    'end': span.end,
                                                    'label': span.label}
                                                   for span in filter_spans(negation_expression_spans)] # Save the negative expressions found
        
        return doc