# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:03:47 2020

@author: jhost
"""

from fuzzyset import OrdinaryFuzzySet, FuzzyVariable
from fuzzysetoperations import AlphaCut, SpecialFuzzySet, StandardIntersection
from sympy import Symbol, Interval, oo # oo is infinity
from pynverse import inversefunc
from sympy import lambdify

# https://www-sciencedirect-com.prox.lib.ncsu.edu/science/article/pii/S0957417412008056

def Unknown():
    formulas = []
    x = Symbol('x')
    formulas.append((1, Interval.Lopen(-oo,55)))
    formulas.append((1 - (x-55)/5, Interval.open(55,60)))
    formulas.append((0, Interval.Ropen(60,oo)))
    return OrdinaryFuzzySet(formulas, 'Unknown')

def Known():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-70)/5, Interval.open(70,75)))
    formulas.append((1, Interval(75,85)))
    formulas.append((1-(x-85)/5, Interval.open(85,90)))
    formulas.append((0, Interval.Lopen(-oo,70)))
    formulas.append((0, Interval.Ropen(90,oo)))
    return OrdinaryFuzzySet(formulas, 'Known')

def UnsatisfactoryUnknown():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-55)/5, Interval.open(55,60)))
    formulas.append((1, Interval(60,70)))
    formulas.append((1-(x-70)/5, Interval.open(70,75)))
    formulas.append((0, Interval.Lopen(-oo,55)))
    formulas.append((0, Interval.Ropen(75, oo)))
    return OrdinaryFuzzySet(formulas, 'Unsatisfactory Unknown')

def Learned():
    formulas = []
    x = Symbol('x')
    formulas.append(((x-85)/5, Interval.open(85,90)))
    formulas.append((1, Interval(90,100)))
    formulas.append((0, Interval.Lopen(-oo,85)))
    return OrdinaryFuzzySet(formulas, 'Learned')

unknown = Unknown()
known = Known()
unsatisfactoryUnknown = UnsatisfactoryUnknown()
learned = Learned()
terms = [unknown, known, unsatisfactoryUnknown, learned]

fuzzyVariable = FuzzyVariable(terms, 'Student Knowledge')
fuzzyVariable.graph(samples=150)


# --- DEMO --- Classify 'x' 

x = 73

z = fuzzyVariable.degree(x)

#alphacut = AlphaCut(known, 0.6, 'AlphaCut')
#alphacut.graph(samples=250)
#specialFuzzySet = SpecialFuzzySet(known, 0.5, 'Special')
#specialFuzzySet.graph()

cuts = []
max_height = 0
idx_of_max = 0
for i in range(len(z)):
    if z[i] > 0:
        specialFuzzySet = SpecialFuzzySet(terms[i], z[i], terms[i].name)
        cuts.append(StandardIntersection([specialFuzzySet,terms[i]]))
        
        # maximum membership principle
        if z[i] > max_height:
            max_height = z[i]
            idx_of_max = i

Z = FuzzyVariable(cuts)
Z.graph()

# maximum membership principle
print('Maximum Membership Principle: %s' % (terms[i].name))