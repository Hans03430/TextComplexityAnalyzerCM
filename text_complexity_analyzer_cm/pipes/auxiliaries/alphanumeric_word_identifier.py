from typing import Iterable, Iterator
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span

from text_complexity_analyzer_cm.utils.utils import is_word
from text_complexity_analyzer_cm.utils.utils import is_content_word


def doc_alpha_words_getter(doc: Doc) -> Iterator:
    '''
    Function that gets the alphabetic words for the entire text.}

    Parameters:
    doc(Doc): Spacy document to be analyzed.

    Yields:
    Iterator: Each alphabetic word
    '''
    for para in doc._.paragraphs:
        for sent in para._.non_empty_sentences:
            for word in sent._.alpha_words:
                yield word

def doc_content_words_getter(doc: Doc) -> Iterator:
    '''
    Function that gets the content words for the entire text.}

    Parameters:
    doc(Doc): Spacy document to be analyzed.

    Yields:
    Iterator: Each content word
    '''
    for para in doc._.paragraphs:
        for sent in para._.non_empty_sentences:
            for word in sent._.content_words:
                yield word    


class AlphanumericWordIdentifier:
    '''
    Pipe that identifies the alphanumeric words (and content words) in a doc. It goes after Paragraphizer.
    '''
    
    name = 'alphanumeric_word_identifier'

    def __init__(self, nlp: Language) -> None:
        '''
        This constructor sets the new extension attributes for Docs.

        It adds a list of alphabetic words to each sentence, as well as the count. It also provides the way to obtain those values for the entire Document/Text

        Parameters:
        nlp(Language): The spacy model that uses this pipeline
        
        Returns:
        None.
        '''
        self._nlp = nlp
        Span.set_extension('alpha_words', default=[]) # List of words for sentence
        Span.set_extension('alpha_words_count', default=0) # Count of words for sentence
        Span.set_extension('content_words', default=[])
        Span.set_extension('content_words_count', default=0)
        Doc.set_extension('content_words', getter=doc_content_words_getter)
        Doc.set_extension('content_words_count', default=0)
        Doc.set_extension('alpha_words', getter=doc_alpha_words_getter)
        Doc.set_extension('alpha_words_count', default=0)

    def __call__(self, doc: Doc) -> Doc:
        '''
        This method will identify the words that are considered alphanumeric.

        Parameters:
        doc(Doc): A Spacy document.

        Returns:
        Doc: The analyzed spacy document.
        '''
        # Find the alphanumeric words for each sentence of the paragraph
        for para in doc._.paragraphs:
            for sent in para._.non_empty_sentences:
                sent._.alpha_words = [token for token in sent if is_word(token)]
                sent._.content_words = [token for token in sent if is_content_word(token)]
                sent._.alpha_words_count = len(sent._.alpha_words)
                sent._.content_words_count = len(sent._.content_words)
                doc._.alpha_words_count += sent._.alpha_words_count
                doc._.content_words_count += sent._.content_words_count
        
        return doc