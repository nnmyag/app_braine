"""
Defines a standard interface for all conductivity models.
Every diffusion→conductivity method must follow this API.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class ConductivityModel(ABC):
    """
    Base class for diffusion → conductivity tensor mapping.

    Input:
        d_evals: [X, Y, Z, 3]
        d_evecs: [X, Y, Z, 3, 3] will be reused for cond. tensor

    Output:
        C       : [X, Y, Z, 3, 3] conductivity tensor
        c_evals : [X, Y, Z, 3] eigenvalues
        c_evecs : [X, Y, Z, 3, 3] eigenvectors
    """

    #: Folder/log/config identifier -- MUST equal the module filename, since
    #: lib_cond.load_cond_model resolves cond_models/<name>.py by it.
    name: str = "base_model"

    #: Compact tag for figure labels, titles and console output, where the full
    #: `name` is too long (e.g. "tractbased_volweighted_legacy"). Declared next to
    #: `name` so adding a model means touching ONE file; read it back from a name
    #: string with lib_cond.model_short_name(). Never use it as a path component --
    #: on-disk layout keys off `name` alone.
    short_name: str = "Base"

    @abstractmethod
    def __call__(
        self,
        d_evals: np.ndarray,
        d_evecs: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform diffusion → conductivity mapping
        """
        pass

    def validate_input(self, d_evals, d_evecs):
        if d_evals.shape[-1] != 3:
            raise ValueError("Diffusion evals must have shape [..., 3]")
        if d_evecs.shape[-2:] != (3, 3):
            raise ValueError("Diffusion evecs must have shape [..., 3, 3]")
        if d_evals.shape[:-1] != d_evecs.shape[:-2]:
            raise ValueError("Mismatch between d_evals and d_evecs spatial shape")