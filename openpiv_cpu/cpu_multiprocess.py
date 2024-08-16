"""This module contains multiprocessing algorithms for a CPU."""

import inspect
from mpire import WorkerPool

class mp_cpu:
    """Multiprocessing class for PIV processing algorithms.
    
    Parameters
    ----------
    func : function
        Function that is multiprocessed.
    n_cpus : int
        Number of processes.
    
    """
    
    def __init__(self, func, n_cpus):
        self.func = func
        assert callable(self.func), "{} must be a callable function.".format("func")
        argspec = inspect.getfullargspec(self.func)
        assert len(argspec.args) == 2 and \
            argspec.varargs is None and \
                argspec.varkw is None, "{} must have the footprint func(pair, index).".format("func")
        
        self.n_cpus = n_cpus
        assert isinstance(self.n_cpus, int) and \
            self.n_cpus > 0, "{} must be a positive int.".format("n_cpus")
    
    def run(self, pairs, indices, progress_bar = True):
        """Computes velocity fields from lists of image pairs and indices.
        
        Parameters
        ----------
        pairs : list
            list of image pairs. Each image pair is a tuple of two 2D arrays containing
            grey levels of two frames.
        indices : list
            list of indices for image pairs.
        
        Returns
        -------
        res : list
            list of results returned by the function.
        
        """
        assert isinstance(pairs, list), "{} must be a list.".format("pairs")
        for pair in pairs:
            assert isinstance(pair, tuple) and len(pair) == 2, \
                "Each item in {} muct be a tuple containing a pair of ndarrays.".format("pairs")
        
        assert isinstance(indices, list) and \
            all(isinstance(item, int) for item in indices) and \
                all(item >= 0 for item in indices), \
                    "{} must be a list of positive {} values.".format("indices", "int")
        
        if self.n_cpus > 1:
            with WorkerPool(n_jobs=self.n_cpus) as pool:
                res = pool.map_unordered(self.func, zip(pairs, indices), progress_bar=progress_bar)

        else:
            res = []
            for pair, index in zip(pairs, indices):
                res.append(self.func(pair, index))
        
        # Sort the results based on indices.
        res = list(zip(res, indices))
        res = sorted(res, key=lambda x: x[1])
        res = [item[0] for item in res]
        return res