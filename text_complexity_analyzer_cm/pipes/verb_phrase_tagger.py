from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from typing import List


def verb_phrases_getter(doc: Doc) -> List[Span]:
    '''
    Function that returns the verb phrases as a list of Spacy Spans.

    Parameters:
    doc(Doc): A spacy doc containing the text.

    Returns:
    List[Span]: A list of spans that represent the verb phrases.
    '''
    return [doc[span['start']:span['end']]
            for span in doc._.verb_phrases_span_indices]

Doc.set_extension('verb_phrases_span_indices', default=[], force=True)
Doc.set_extension('verb_phrases', force=True, getter=verb_phrases_getter)

class VerbPhraseTagger:
    '''
    This tagger has the task to find all verb phrases in a document. It needs to go after the 'Parser' pipeline component.
    '''
    name = 'verb phrase tagger'

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

        if language == 'es': # Verb phrases for spanish
            self._pattern = [{'POS': {'IN': ['AUX', 'VERB']}, 'OP': '+'},
                             {'POS': {'IN': ['ADP', 'SCONJ', 'CONJ', 'INTJ']}, 'OP': '*'},
                             {'POS': 'ADP', 'TAG': 'ADP__AdpType=Prep', 'OP': '*'},
                             {'POS': {'IN': ['AUX', 'VERB']}}] # The pattern for verb phrases in spanish
        else: # Support for future languages
            pass

        self._matcher.add('verb phrase', None, self._pattern) # Add the verb phrase pattern

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all verb phrases and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        matches = self._matcher(doc)
        verb_phrase_spans = [doc[start:end] for _, start, end in matches]

        doc._.verb_phrases_span_indices = [{'start': span.start,
                                            'end': span.end,
                                            'label': span.label}
                                           for span in filter_spans(verb_phrase_spans)] # Save the noun phrases found

        return doc