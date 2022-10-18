from spacy.lang.es import Spanish
from spacy.language import Language

from text_complexity_analyzer_cm.pipes.auxiliaries.alphanumeric_word_identifier import AlphanumericWordIdentifier
from text_complexity_analyzer_cm.pipes.auxiliaries.paragraphizer import Paragraphizer
from text_complexity_analyzer_cm.pipes.auxiliaries.syllablelizer import Syllablelizer
from text_complexity_analyzer_cm.pipes.coh_metrix_indices.descriptive_indices import DescriptiveIndices

@Spanish.factory('alphanumeric_word_identifier')
def create_es_alphanumeric_word_identifier(nlp: Language, name: str) -> AlphanumericWordIdentifier:
    '''
    Function that creates an alphanumeric word identifier for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    AlphanumericWordIdentifier: The pipe that finds alphanumeric words.
    '''
    return AlphanumericWordIdentifier(nlp)

@Spanish.factory('paragraphizer', default_config={'paragraph_delimiter': '\n\n'})
def create_es_paragraphizer(nlp: Language, name: str, paragraph_delimiter: str) -> Paragraphizer:
    '''
    Function that creates a paragraph splitter for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.
    paragraph_delimiter(str): Character used to split the paragraphs.

    Returns:
    Paragraphizer: The pipe that finds alphanumeric words.
    '''
    return Paragraphizer(nlp, paragraph_delimiter)

@Spanish.factory('syllablelizer', default_config={'language': 'es'})
def create_es_syllablelizer(nlp: Language, name: str, language: str) -> Syllablelizer:
    '''
    Function that creates a syllable splitter for spanish.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.
    language(str): Character used to split the paragraphs

    Returns:
    Syllablelizer: The pipe that finds alphanumeric words.
    '''
    return Syllablelizer(nlp, language)

@Spanish.factory('descriptive_indices')
def create_es_descriptive_indices(nlp: Language, name: str) -> DescriptiveIndices:
    '''
    Function that creates descriptive indices pipe.
    
    Paramters:
    nlp(Language): Spacy model that will be used for the pipeline.
    name(str): Name of the pipe.

    Returns:
    ParagraphSplitter: The pipe that finds alphanumeric words.
    '''
    return DescriptiveIndices(nlp)