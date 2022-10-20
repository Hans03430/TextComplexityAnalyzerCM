from spacy.language import Language
from text_complexity_analyzer_cm.pipes.auxiliaries.wrapper_serializer import WrapperSerializer

@Language.factory('wrapper_serializer')
def create_wrapper_serializer(nlp: Language, name: str) -> WrapperSerializer:
    '''
    Function that creates a wrapper serializer pipe.

    Parameters:
    nlp(Language): Language model to use.
    name(str): Name of the pipe.

    Returns:
    WrapperSerializer: Pipe that wraps all entities of a Doc, and the doc itself, to prepare it for serialization.
    '''
    return WrapperSerializer(nlp)