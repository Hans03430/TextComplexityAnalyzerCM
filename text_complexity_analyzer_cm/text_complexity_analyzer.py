import pickle
import spacy
import time

from text_complexity_analyzer_cm.constants import ACCEPTED_LANGUAGES
from text_complexity_analyzer_cm.constants import BASE_DIRECTORY
from text_complexity_analyzer_cm.coh_metrix_indices.connective_indices import ConnectiveIndices
from text_complexity_analyzer_cm.coh_metrix_indices.descriptive_indices import DescriptiveIndices
from text_complexity_analyzer_cm.coh_metrix_indices.lexical_diversity_indices import LexicalDiversityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.readability_indices import ReadabilityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.referential_cohesion_indices import ReferentialCohesionIndices
from text_complexity_analyzer_cm.coh_metrix_indices.syntactic_complexity_indices import SyntacticComplexityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.syntactic_pattern_density_indices import SyntacticPatternDensityIndices
from text_complexity_analyzer_cm.coh_metrix_indices.word_information_indices import WordInformationIndices
from text_complexity_analyzer_cm.pipes.negative_expression_tagger import NegativeExpressionTagger
from text_complexity_analyzer_cm.pipes.noun_phrase_tagger import NounPhraseTagger
from text_complexity_analyzer_cm.pipes.syllable_splitter import SyllableSplitter
from text_complexity_analyzer_cm.pipes.verb_phrase_tagger import VerbPhraseTagger
from text_complexity_analyzer_cm.pipes.causal_connectives_tagger import CausalConnectivesTagger
from text_complexity_analyzer_cm.pipes.logical_connectives_tagger import LogicalConnectivesTagger
from text_complexity_analyzer_cm.pipes.adversative_connectives_tagger import AdversativeConnectivesTagger
from text_complexity_analyzer_cm.pipes.temporal_connectives_tagger import TemporalConnectivesTagger
from text_complexity_analyzer_cm.pipes.additive_connectives_tagger import AdditiveConnectivesTagger
from text_complexity_analyzer_cm.pipes.referential_cohesion_adjacent_sentences_analyzer import ReferentialCohesionAdjacentSentencesAnalyzer
from text_complexity_analyzer_cm.pipes.referential_cohesion_all_sentences_analyzer import ReferentialCohesionAllSentencesAnalyzer
from text_complexity_analyzer_cm.pipes.feature_counter import FeatureCounter
from typing import Dict
from typing import List
from typing import Tuple


class TextComplexityAnalyzer:
    '''
    This class groups all of the indices in order to calculate them in one go. It works for a specific language.

    To use this class, instantiate an object with it. For example:
    tca = TextComplexityAnalyzer('es')

    Notice that a short version of the language was passed. The only languages available for now are: 'es'.

    To calculate the implemented coh-metrix indices for a text, do the following:
    m1, m2, m3, m4, m5, m6, m7, m8 = tca.calculate_all_indices_for_one_text(text='Example text', workers=-1)

    Here, all available cores will be used to analyze the text passed as parameter.

    To predict the category of a text, do the following:
    prediction = tca.predict_text_category(text='Example text', workers=-1)

    The example uses the default classifier stored along the library.
    '''
    def __init__(self, language:str = 'es') -> None:
        '''
        This constructor initializes the analizer for a specific language.

        Parameters:
        language(str): The language that the texts are in.
        
        Returns:
        None.
        '''
        if not language in ACCEPTED_LANGUAGES:
            raise ValueError(f'Language {language} is not supported yet')
        
        self.language = language
        self._nlp = spacy.load(ACCEPTED_LANGUAGES[language], disable=['ner'])
        self._nlp.max_length = 3000000
        self._nlp.add_pipe(self._nlp.create_pipe('sentencizer'))
        self._nlp.add_pipe(SyllableSplitter(language), after='tagger')
        self._nlp.add_pipe(NounPhraseTagger(language), after='parser')
        self._nlp.add_pipe(VerbPhraseTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(NegativeExpressionTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(CausalConnectivesTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(LogicalConnectivesTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(AdversativeConnectivesTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(TemporalConnectivesTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(AdditiveConnectivesTagger(self._nlp, language), after='tagger')
        self._nlp.add_pipe(ReferentialCohesionAdjacentSentencesAnalyzer(language), after='sentencizer')
        self._nlp.add_pipe(ReferentialCohesionAllSentencesAnalyzer(language), after='sentencizer')
        self._nlp.add_pipe(FeatureCounter(language), last=True)
        self._di = DescriptiveIndices(language=language, nlp=self._nlp)
        self._spdi = SyntacticPatternDensityIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)
        self._wii = WordInformationIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)
        self._sci = SyntacticComplexityIndices(language=language, nlp=self._nlp)
        self._ci = ConnectiveIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)
        self._ldi = LexicalDiversityIndices(language=language, nlp=self._nlp)
        self._ri = ReadabilityIndices(language=language, nlp=self._nlp, descriptive_indices=self._di)
        self._rci = ReferentialCohesionIndices(language=language, nlp=self._nlp)

        # Load default classifier
        self._classifier = pickle.load(open(f'{BASE_DIRECTORY}/model/classifier.pkl', 'rb'))
        self._scaler = pickle.load(open(f'{BASE_DIRECTORY}/model/scaler.pkl', 'rb'))
        self._indices = ['CNCADC', 'CNCAdd', 'CNCAll', 'CNCCaus', 'CNCLogic', 'CNCTemp', 'CRFANP1', 'CRFANPa', 'CRFAO1', 'CRFAOa', 'CRFCWO1', 'CRFCWO1d', 'CRFCWOa', 'CRFCWOad', 'CRFNO1', 'CRFNOa', 'CRFSO1', 'CRFSOa', 'DESPC', 'DESPL', 'DESPLd', 'DESSC', 'DESSL', 'DESSLd', 'DESWC', 'DESWLlt', 'DESWLltd', 'DESWLsy', 'DESWLsyd', 'DRNEG', 'DRNP', 'DRVP', 'LDTTRa', 'LDTTRcw', 'RDFHGL', 'SYNLE', 'SYNNP', 'WRDADJ', 'WRDADV', 'WRDNOUN', 'WRDPRO', 'WRDPRP1p', 'WRDPRP1s', 'WRDPRP2p', 'WRDPRP2s', 'WRDPRP3p', 'WRDPRP3s', 'WRDVERB']


    def calculate_descriptive_indices_for_one_text(self, text: str, workers: int=-1) -> Dict:
        '''
        This method calculates the descriptive indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        Dict: The dictionary with the descriptive indices.
        '''
        indices = {}
        indices['DESPC'] = self._di.get_paragraph_count_from_text(text=text)
        indices['DESSC'] = self._di.get_sentence_count_from_text(text=text, workers=workers)
        indices['DESWC'] = self._di.get_word_count_from_text(text=text, workers=workers)
        length_of_paragraph = self._di.get_length_of_paragraphs(text=text, workers=workers)
        indices['DESPL'] = length_of_paragraph.mean
        indices['DESPLd'] = length_of_paragraph.std
        length_of_sentences = self._di.get_length_of_sentences(text=text, workers=workers)
        indices['DESSL'] = length_of_sentences.mean
        indices['DESSLd'] = length_of_sentences.std
        syllables_per_word = self._di.get_syllables_per_word(text=text, workers=workers)
        indices['DESWLsy'] = syllables_per_word.mean
        indices['DESWLsyd'] = syllables_per_word.std
        length_of_words = self._di.get_length_of_words(text=text, workers=workers)
        indices['DESWLlt'] = length_of_words.mean
        indices['DESWLltd'] = length_of_words.std
        return indices

    def calculate_word_information_indices_for_one_text(self, text: str, workers: int=-1, word_count: int=None) -> Dict:
        '''
        This method calculates the descriptive indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        word_count(int): The amount of words that the current text has in order to calculate the incidence.

        Returns:
        Dict: The dictionary with the word information indices.
        '''
        indices = {}
        indices['WRDNOUN'] = self._wii.get_noun_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDVERB'] = self._wii.get_verb_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDADJ'] = self._wii.get_adjective_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDADV'] = self._wii.get_adverb_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRO'] = self._wii.get_personal_pronoun_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP1s'] = self._wii.get_personal_pronoun_first_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP1p'] = self._wii.get_personal_pronoun_first_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP2s'] = self._wii.get_personal_pronoun_second_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP2p'] = self._wii.get_personal_pronoun_second_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP3s'] = self._wii.get_personal_pronoun_third_person_singular_form_incidence(text=text, workers=workers, word_count=word_count)
        indices['WRDPRP3p'] = self._wii.get_personal_pronoun_third_person_plural_form_incidence(text=text, workers=workers, word_count=word_count)
        
        return indices

    def calculate_syntactic_pattern_density_indices_for_one_text(self, text: str, workers: int=-1, word_count: int=None) -> Dict:
        '''
        This method calculates the syntactic pattern indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        word_count(int): The amount of words that the current text has in order to calculate the incidence.

        Returns:
        Dict: The dictionary with the syntactic pattern indices.
        '''
        indices = {}
        indices['DRNP'] = self._spdi.get_noun_phrase_density(text=text, workers=workers, word_count=word_count)
        indices['DRVP'] = self._spdi.get_verb_phrase_density(text=text, workers=workers, word_count=word_count)
        indices['DRNEG'] = self._spdi.get_negation_expressions_density(text=text, workers=workers, word_count=word_count)

        return indices
        
    def calculate_syntactic_complexity_indices_for_one_text(self, text: str, workers: int=-1) -> Dict:
        '''
        This method calculates the syntactic complexity indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        Dict: The dictionary with the syntactic complexity indices.
        '''
        indices = {}
        indices['SYNNP'] = self._sci.get_mean_number_of_modifiers_per_noun_phrase(text=text, workers=workers)
        indices['SYNLE'] = self._sci.get_mean_number_of_words_before_main_verb(text=text, workers=workers)

        return indices

    def calculate_connective_indices_for_one_text(self, text: str, workers: int=-1, word_count: int=None) -> Dict:
        '''
        This method calculates the connectives indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        word_count(int): The amount of words that the current text has in order to calculate the incidence.

        Returns:
        Dict: The dictionary with the connectives indices.
        '''
        indices = {}
        indices['CNCAll'] = self._ci.get_all_connectives_incidence(text=text, workers=workers, word_count=word_count)
        indices['CNCCaus'] = self._ci.get_causal_connectives_incidence(text=text, workers=workers, word_count=word_count)
        indices['CNCLogic'] = self._ci.get_logical_connectives_incidence(text=text, workers=workers, word_count=word_count)
        indices['CNCADC'] = self._ci.get_adversative_connectives_incidence(text=text, workers=workers, word_count=word_count)
        indices['CNCTemp'] = self._ci.get_temporal_connectives_incidence(text=text, workers=workers, word_count=word_count)
        indices['CNCAdd'] = self._ci.get_additive_connectives_incidence(text=text, workers=workers, word_count=word_count)

        return indices

    def calculate_lexical_diversity_indices_for_one_text(self, text: str, workers: int=-1) -> Dict:
        '''
        This method calculates the lexical diversity indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        word_count(int): The amount of words that the current text has in order to calculate the incidence.

        Returns:
        Dict: The dictionary with the lexical diversity indices.
        '''
        indices = {}
        indices['LDTTRa'] = self._ldi.get_type_token_ratio_between_all_words(text=text, workers=workers)
        indices['LDTTRcw'] = self._ldi.get_type_token_ratio_of_content_words(text=text, workers=workers)

        return indices

    def calculate_readability_indices_for_one_text(self, text: str, workers: int=-1, mean_syllables_per_word: int=None, mean_words_per_sentence: int=None) -> Dict:
        '''
        This method calculates the readability indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        mean_syllables_per_word(int): The mean of syllables per word in the text.
        mean_words_per_sentence(int): The mean amount of words per sentences in the text.

        Returns:
        Dict: The dictionary with the readability indices.
        '''
        indices = {}
        
        if self.language == 'es':
            indices['RDFHGL'] = self._ri.calculate_fernandez_huertas_grade_level(text=text, workers=workers, mean_words_per_sentence=mean_words_per_sentence, mean_syllables_per_word=mean_syllables_per_word)

        return indices

    def calculate_referential_cohesion_indices_for_one_text(self, text: str, workers: int=-1) -> Dict:
        '''
        This method calculates the referential cohesion indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        Dict: The dictionary with the readability indices.
        '''
        indices = {}
        indices['CRFNO1'] = self._rci.get_noun_overlap_adjacent_sentences(text=text, workers=workers)
        indices['CRFNOa'] = self._rci.get_noun_overlap_all_sentences(text=text, workers=workers)
        indices['CRFAO1'] = self._rci.get_argument_overlap_adjacent_sentences(text=text, workers=workers)
        indices['CRFAOa'] = self._rci.get_argument_overlap_all_sentences(text=text, workers=workers)
        indices['CRFSO1'] = self._rci.get_stem_overlap_adjacent_sentences(text=text, workers=workers)
        indices['CRFSOa'] = self._rci.get_stem_overlap_all_sentences(text=text, workers=workers)
        content_word_overlap_adjacent = self._rci.get_content_word_overlap_adjacent_sentences(text=text, workers=workers)
        indices['CRFCWO1'] = content_word_overlap_adjacent.mean
        indices['CRFCWO1d'] = content_word_overlap_adjacent.std
        content_word_overlap_all = self._rci.get_content_word_overlap_all_sentences(text=text, workers=workers)
        indices['CRFCWOa'] = content_word_overlap_all.mean
        indices['CRFCWOad'] = content_word_overlap_all.std
        indices['CRFANP1'] = self._rci.get_anaphore_overlap_adjacent_sentences(text=text, workers=workers)
        indices['CRFANPa'] = self._rci.get_anaphore_overlap_all_sentences(text=text, workers=workers)

        return indices

    def calculate_all_indices_for_one_text(self, text: str, workers: int=-1) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict]:
        '''
        This method calculates the referential cohesion indices and stores them in a dictionary.

        Parameters:
        text(str): The text to be analyzed.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.

        Returns:
        (Dict, Dict, Dict, Dict, Dict, Dict, Dict, Dict): The dictionary with the all the indices.
        '''
        if workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0.')
        else:
            start = time.time()
            descriptive = self.calculate_descriptive_indices_for_one_text(text=text, workers=workers)
            word_count = descriptive['DESWC']                
            mean_words_per_sentence = descriptive['DESSL']
            mean_syllables_per_word = descriptive['DESWLsy']
            word_information = self.calculate_word_information_indices_for_one_text(text=text, workers=workers, word_count=word_count)
            syntactic_pattern = self.calculate_syntactic_pattern_density_indices_for_one_text(text=text, workers=workers, word_count=word_count)
            syntactic_complexity = self.calculate_syntactic_complexity_indices_for_one_text(text=text, workers=workers)
            connective = self.calculate_connective_indices_for_one_text(text=text, workers=workers, word_count=word_count)
            lexical_diversity = self.calculate_lexical_diversity_indices_for_one_text(text=text, workers=workers)
            readability = self.calculate_readability_indices_for_one_text(text, workers=workers, mean_words_per_sentence=mean_words_per_sentence, mean_syllables_per_word=mean_syllables_per_word)
            referential_cohesion = self.calculate_referential_cohesion_indices_for_one_text(text=text, workers=workers)
            end = time.time()
            print(f'Text analyzed in {end - start} seconds.')

            return descriptive, word_information, syntactic_pattern, syntactic_complexity, connective, lexical_diversity, readability, referential_cohesion

    def predict_text_category(self, text: str, workers: int=-1, classifier=None, scaler=None, indices: List=None) -> int:
        '''
        This method receives a text and predict its category based on the classification model trained.

        Parameters:
        text(str): The text to predict its category.
        workers(int): Amount of threads that will complete this operation. If it's -1 then all cpu cores will be used.
        classifier: Optional. A supervised learning model that implements the 'predict' method. If None, the default classifier is used.
        scaler: Optional. A object that implements the 'transform' method that scales the indices of the text to analyze. It must be the same as the one used in the classifier, if a scaler was used. Pass None if no scaler was used during the custom classifier's training.
        indices(List): Optional. Ignored if the default classifier is used. The name indices which the classifier was trained with. They must be in the same order as the ones that were used at training and also be the same. 

        Returns:
        int: The category of the text represented as a number
        '''
        if workers == 0 or workers < -1:
            raise ValueError('Workers must be -1 or any positive number greater than 0.')
        if classifier is not None and not hasattr(classifier, 'predict'):
            raise ValueError('The custom surpervised learning model (classifier) must have the \'predict\' method.d')
        if classifier is not None and indices is None:
            raise ValueError('You must provide the names of the metrics used to train the custom classifier in the same order and amount that they were at the time of training said classifier.')
        if classifier is not None and scaler is not None and not hasattr(scaler, 'transform'):
            raise ValueError('The custom scaling model (scaler) for the custom classifier must have the \'transform\' method.')
        else:
            descriptive, word_information, syntactic_pattern, syntactic_complexity, connective, lexical_diversity, readability, referential_cohesion = self.calculate_all_indices_for_one_text(text, workers)
            metrics = {**descriptive, **word_information, **syntactic_pattern, **syntactic_complexity, **connective, **lexical_diversity, **readability, **referential_cohesion}
            if classifier is None: # Default indices
                indices_values = [[metrics[key] for key in self._indices]]

                return self._classifier.predict(self._scaler.transform(indices_values))
            else: # Indices used by the custom classifier
                indices_values = [[metrics[key] for key in indices]]
                    
                return list(classifier.predict(indices_values if scaler is None else scaler.transform(indices_values)))
