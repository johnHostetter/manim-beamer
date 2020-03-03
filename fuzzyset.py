# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:40:22 2020

@author: jhost
"""
import sympy
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt

# https://docs.sympy.org/latest/modules/integrals/integrals.html
# https://docs.sympy.org/latest/modules/sets.html
# https://numpydoc.readthedocs.io/en/latest/example.html

class FuzzySet:
    """
    A parent class for all fuzzy sets to inherit. Allows the user to visualize the fuzzy set.
    
    Methods
    -------
    graph(lower=0, upper=100, samples=100)
        Graphs the fuzzy set in the universe of elements.
    """
    
    def __init__(self):
        """
        The empty constructor.
        """
        pass
    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the fuzzy set in the universe of elements.
        
        Parameters
        ----------
        lower : 'float', optional
            Default value is 0. Specifies the infimum x value for the graph.
        upper : 'float', optional
            Default value is 100. Specifies the supremum x value for the graph.
        samples : 'int', optional
            Default value is 100. Specifies the number of x values to test in the domain
            to approximate the grpah. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.        
        """
        x_list = np.linspace(lower, upper, samples)
        y_list = []
        for x in x_list:
            y_list.append(self.degree(x))
        if self.name != None:    
            plt.title('%s Fuzzy Set' % self.name)
        else:
            plt.title('Unnamed Fuzzy Set')
        
        plt.axes()
        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.plot(x_list, y_list, color='grey', label='mu')
        plt.legend()
        plt.show()
    
class OrdinaryFuzzySet(FuzzySet):
    """
    An ordinary fuzzy set that is of type 1 and level 1.
    
    Attributes
    ----------
    formulas : 'list'
        A list of 2-tuples. The first element in the tuple at index 0 is the formula 
        equal to f(x) and the second element in the tuple at index 1 is the Interval 
        where the formula in the tuple is valid.
        
        Warning: Formulas should be organized in the list such that the formulas and
        their corresponding intervals are specified from the smallest possible x values
        to the largest possible x values.
        
        The list of formulas provided constitutes the piece-wise function of the
        fuzzy set's membership function.
    name : 'str'/'None'
        Default value is None. Allows the user to specify the name of the fuzzy set.
        This feature is useful when visualizing the fuzzy set, and its interaction with
        other fuzzy sets in the same space.
    
    Methods
    -------
    fetch(x)
        Calculates the corresponding formula for the provided x value where x is a(n) int/float.
    degree(x)
        Calculates the degree of membership for the provided x value where x is a(n) int/float.
    height()
        Calculates the height of the fuzzy set.
    graph(lower=0, upper=100, samples=100)
        Graphs the fuzzy set in the universe of elements.
        
    Examples
    --------
        >>> formulas = []
        >>> x = Symbol('x')
        >>> formulas.append((1, Interval.Lopen(-oo,20)))
        >>> formulas.append(((35-x)/15,Interval.open(20,35)))
        >>> formulas.append((0, Interval.Ropen(35,oo)))
        >>> OrdinaryFuzzySet(formulas, 'A1')
    """
    
    def __init__(self, formulas, name=None):
        """                   
        Parameters
        ----------
        formulas : 'list'
            A list of 2-tuples. The first element in the tuple at index 0 is the formula 
            equal to f(x) and the second element in the tuple at index 1 is the Interval 
            where the formula in the tuple is valid.
            
            Warning: Formulas should be organized in the list such that the formulas and
            their corresponding intervals are specified from the smallest possible x values
            to the largest possible x values.
            
            The list of formulas provided constitutes the piece-wise function of the
            fuzzy set's membership function.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        FuzzySet.__init__(self)
        self.formulas = formulas
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
        formula = self.fetch(x)[0]
        try:
            y = float(formula.subs(Symbol('x'), x))
        except AttributeError:
            y = formula
        return y
    def height(self):
        """
        Calculates the height of the fuzzy set.
        
        Returns
        -------
        height : 'float'
            The height, or supremum, of the fuzzy set.
        """
        heights = []
        for formula in self.formulas:
            if isinstance(formula[0], sympy.Expr):
                inf_x = formula[1].inf
                sup_x = formula[1].sup
                if formula[1].left_open:
                    inf_x += 1e-8
                if formula[1].right_open:
                    sup_x -= 1e-8
                inf_y = formula[0].subs(Symbol('x'), inf_x)
                sup_y = formula[0].subs(Symbol('x'), sup_x)
                heights.append(inf_y)
                heights.append(sup_y)
            else:
                heights.append(formula[0])
        return max(heights)
    
class FuzzyVariable(FuzzySet):
    """
    A fuzzy variable, or linguistic variable, that contains fuzzy sets.
    
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
    
    def __init__(self, fuzzySets, name=None):
        """        
        Parameters
        ----------
        fuzzySets : 'list'
            A list of elements each of type OrdinaryFuzzySet.
        name : 'str'/'None'
            Default value is None. Allows the user to specify the name of the fuzzy set.
            This feature is useful when visualizing the fuzzy set, and its interaction with
            other fuzzy sets in the same space.
        """
        FuzzySet.__init__(self)
        self.fuzzySets = fuzzySets
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
        for fuzzySet in self.fuzzySets:
            degrees.append(fuzzySet.degree(x))
        return tuple(degrees)
    def graph(self, lower=0, upper=100, samples=100):
        """
        Graphs the fuzzy set in the universe of elements.
        
        Parameters
        ----------
        lower : 'float', optional
            Default value is 0. Specifies the infimum x value for the graph.
        upper : 'float', optional
            Default value is 100. Specifies the supremum x value for the graph.
        samples : 'int', optional
            Default value is 100. Specifies the number of x values to test in the domain
            to approximate the grpah. A higher sample value will yield a higher resolution
            of the graph, but large values will lead to performance issues.        
        """
        for fuzzySet in self.fuzzySets:
            x_list = np.linspace(lower, upper, samples)
            y_list = []
            for x in x_list:
                y_list.append(fuzzySet.degree(x))
            plt.plot(x_list, y_list, color=np.random.rand(3,), label=fuzzySet.name)

        if self.name != None:    
            plt.title('%s Fuzzy Variable' % self.name)
        else:
            plt.title('Unnamed Fuzzy Variable')
        
        plt.axes()
        plt.xlim([lower, upper])
        plt.ylim([0, 1.1])
        plt.xlabel('Elements of Universe')
        plt.ylabel('Degree of Membership')
        plt.legend()
        plt.show()