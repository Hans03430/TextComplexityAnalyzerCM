from dataclasses import dataclass

@dataclass
class StatisticsResults:
    '''
    This data class stores the mean and standard deviation.
    '''
    mean: float=0
    std: float=0