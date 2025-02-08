import numpy as np

class RepeatedValuesException(Exception):
    """Exception raised for repeated values in the path."""
    def __init__(self, message="Path contains repeated values"):
        self.message = message
        super().__init__(self.message)
    
    
class SanityAssertions:
    @staticmethod
    def assert_no_repeated_values(path: np.ndarray):
        """Check if the path contains repeated values and raise an exception if it does."""
        unique_values = np.unique(path)
        if len(unique_values) != len(path):
            raise RepeatedValuesException()