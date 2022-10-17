from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.util import filter_spans

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES

temporal_connectives_getter = lambda doc: [doc[span['start']:span['end']]
                                           for span in doc._.temporal_connectives_span_indices]

Doc.set_extension('temporal_connectives_span_indices', force=False, default=[])
Doc.set_extension('temporal_connectives', force=False, getter=temporal_connectives_getter)

class TemporalConnectivesTagger:
    '''
    This tagger has the task to find all temporal connectives in a document. It needs to go after the 'Tagger' pipeline component.
    '''
    name = 'temporal connective tagger'

    def __init__(self, nlp, language: str='es') -> None:
        '''
        This constructor will initialize the object that tags temporal connectives.

        Parameters:
        nlp: The Spacy model to use this tagger with.
        language: The language that this pipeline will be used in.

        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')

        self._language = language
        self._matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
        self._connectives = []
        if language == 'es': # Temporal connectives for spanish
            self._connectives = ['actualmente', 'ahora', 'después', 'más tarde', 'más adelante', 'a continuación', 'antes', 'mientras', 'érase una vez', 'hace mucho tiempo', 'tiempo antes', 'finalmente', 'inicialmente', 'ya', 'simultáneamente', 'previamente', 'anteriormente', 'posteriormente', 'al mismo tiempo', 'durante']
        else: # Support for future languages
            pass

        for con in self._connectives:
            self._matcher.add(con, None, nlp(con))
        

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will find all temporal connectives and store them in an iterable.

        Parameters:
        doc(Doc): A Spacy document.
        '''
        matches = self._matcher(doc)
        temporal_connectives_spans = [doc[start:end] for _, start, end in matches]

        doc._.temporal_connectives_span_indices = [{'start': span.start,
                                                    'end': span.end,
                                                    'label': span.label}
                                                   for span in filter_spans(temporal_connectives_spans)] # Save the temporal connectives found
        
        return doc