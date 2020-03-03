# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:49:59 2020

@author: jhost
"""
import sympy
from fuzzyset import FuzzySet
from pynverse import inversefunc
from sympy import lambdify, Symbol, Interval, Union

class SpecialFuzzySet(FuzzySet):
    """ 
    The special fuzzy set membership function for a given element x in the universe of 
    discourse X, is defined as the alpha value multipled by the element x's degree of 
    membership within the fuzzy set's alpha cut.
    
    Attributes
    ----------
    fuzzyset : 'OrdinaryFuzzySet'
        An ordinary fuzzy set to retrieve the special fuzzy set given the alpha.
    alpha : 'float'
        The alpha value that elements' membership degree must exceed or be equal to.
    name : 'str'/'None'
        Default value is None. Allows the user to specify the name of the fuzzy set.
        This feature is useful when visualizing the fuzzy set, and its interaction with
        other fuzzy sets in the same space.
        
    Methods
    -------
    fetch(x)
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.
    degree(x)
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
    height()
        Calculates the height of the special fuzzy set.
    """
    
    def __init__(self, fuzzyset, alpha, name=None):
        """
        Parameters
        ----------
            fuzzyset : 'OrdinaryFuzzySet'
                An ordinary fuzzy set to retrieve the special fuzzy set given the alpha.
            alpha : 'float'
                The alpha value that elements' membership degree must exceed or be equal to.
            name : 'str'/'None'
                Default value is None. Allows the user to specify the name of the fuzzy set.
                This feature is useful when visualizing the fuzzy set, and its interaction with
                other fuzzy sets in the same space.
        """
        alphaCut = AlphaCut(fuzzyset, alpha)
        interval = alphaCut.formulas[0][1]
        for formula in alphaCut.formulas[1:]:
            interval = Union(interval, formula[1])
        self.formulas = [(alpha, interval)]
        self.alpha = alpha
        self.name = name
    def fetch(self, x):
        """ 
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding Interval. Returns
            None if a formula for the element x could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(x): # check the formula's interval to see if it contains x
                return formula
        return None
    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        result = self.fetch(x)
        if result != None:
            formula = result[0]
        else:
            return 0
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y  
    def height(self):
        """
        Calculates the height of the special fuzzy set.
        
        Returns
        -------
        height : 'float'
            The height, or supremum, of the Special Fuzzy Set.
        """
        return self.alpha

class AlphaCut(FuzzySet):
    """ 
    The alpha cut of a fuzzy set yields a crisp set.
    
    Attributes
    ----------
    formulas : 'list'
        A list of 2-tuples. The first element in the tuple at index 0 is the formula 
        equal to f(x) and the second element in the tuple at index 1 is the Interval 
        where the formula in the tuple is valid.
    alpha : 'float'
        The alpha value that elements' membership degree must exceed or be equal to.
    name : 'str'/'None'
        Default value is None. Allows the user to specify the name of the fuzzy set.
        This feature is useful when visualizing the fuzzy set, and its interaction with
        other fuzzy sets in the same space.
    
    Methods
    -------
    fetch(x)
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.
    degree(x)
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
    """
    
    def __init__(self, fuzzyset, alpha, name=None):
        """
        Parameters
        ----------
        formulas : 'list'
            A list of 2-tuples. The first element in the tuple at index 0 is the formula 
            equal to f(x) and the second element in the tuple at index 1 is the Interval 
            where the formula in the tuple is valid.
        alpha : 'float'
            The alpha value that elements' membership degree must exceed or be equal to.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        self.alpha = alpha
        self.name = name
        formulas = []
        for formula in fuzzyset.formulas:
            if isinstance(formula[0], sympy.Expr):
                x = inversefunc(lambdify(Symbol('x'), formula[0], 'numpy'), y_values=alpha)
                if formula[1].contains(x):
                    # the x is within the interval, now check the direction
                    y = formula[0].subs(Symbol('x'), x - (1e-6))
                    if y >= alpha:
                        # then all values less than or equal to x are valid
                        if formula[1].left_open:    
                            interval = Interval.Lopen(formula[1].inf, x)
                        else:
                            interval = Interval(formula[1].inf, x)
                    else:
                        # then all values greater than or equal to x are valid
                        if formula[1].right_open:
                            interval = Interval.Ropen(x, formula[1].sup)
                        else:
                            interval = Interval(x, formula[1].sup)
                    formula = list(formula)
                    formula[1] = interval
                    formula = tuple(formula)
                    formulas.append(formula)
            else:
                if formula[0] >= alpha:
                    formulas.append(formula)
        self.formulas = formulas
    def fetch(self, x):
        """ 
        Fetch the corresponding formula for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        formula : 'tuple'/'None'
            Returns the tuple containing the formula and corresponding interval. Returns
            None if a formula for the element x could not be found.
        """
        for formula in self.formulas:
            if formula[1].contains(x): # check the formula's interval to see if it contains x
                return formula
        return None
    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        result = self.fetch(x)
        if result != None:
            formula = result[0]
        else:
            return 0
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y
        
def StandardComplement(fuzzySet):
    """
    Obtains the standard complement of a fuzzy set as defined by Lotfi A. Zadeh.
    
    Returns True if successful, else returns False.
    
    Parameters
    ----------
    fuzzySet : 'OrdinaryFuzzySet'
    
    Returns
    -------
    success : 'bool'
    """
    
    if isinstance(fuzzySet, FuzzySet):
        formulas = []
        for formula in fuzzySet.formulas:
            formula = list(formula)
            formula[0] = 1- formula[0]
            formula = tuple(formula)
            formulas.append(formula)
        fuzzySet.formulas = formulas
        return True
    return False

class StandardUnion(FuzzySet):
    """
    A standard union of one or more ordinary fuzzy sets.
    
    Attributes
    ----------
    fuzzySets : 'list'
        A list of elements each of type OrdinaryFuzzySet.
    name : 'str'/'None'
        Default value is None. Allows the user to specify the name of the fuzzy set.
        This feature is useful when visualizing the fuzzy set, and its interaction with
        other fuzzy fets in the same space.
    
    Methods
    -------
    degree(x)
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
    graph(lower=0, upper=100, samples=100)
        Graphs the fuzzy set in the universe of elements.
    """
    
    def __init__(self, fuzzysets, name=None):
        """        
        Parameters
        ----------
        fuzzySets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        FuzzySet.__init__(self)
        self.fuzzysets = fuzzysets
        self.name = name
    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        degrees = []
        for fuzzyset in self.fuzzysets:
            degrees.append(fuzzyset.degree(x))
        return max(degrees)
        
class StandardIntersection(FuzzySet):
    """
    A standard intersection of one or more ordinary fuzzy sets.
    
    Attributes
    ----------
    fuzzySets : 'list'
        A list of elements each of type OrdinaryFuzzySet.
    name : 'str'/'None'
        Default value is None. Allows the user to specify the name of the fuzzy set.
        This feature is useful when visualizing the fuzzy set, and its interaction with
        other fuzzy fets in the same space.
    
    Methods
    -------
    degree(x)
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
    graph(lower=0, upper=100, samples=100)
        Graphs the fuzzy set in the universe of elements.
    """
    
    def __init__(self, fuzzysets, name=None):
        """        
        Parameters
        ----------
        fuzzySets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy fets in the same space.
        """
        FuzzySet.__init__(self)
        self.fuzzysets = fuzzysets
        self.name = name
    def degree(self, x):
        """
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
        
        Parameters
        ----------
        x : 'float'
            The parameter x is the element from the universe of discourse X.
        
        Returns
        -------
        y : 'float'
            The degree of membership for element x.
        """
        degrees = []
        for fuzzyset in self.fuzzysets:
            degrees.append(fuzzyset.degree(x))
        return min(degrees)