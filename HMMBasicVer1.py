
# coding: utf-8

# In[29]:

#Theory Behind HMM:
""" A Markov chain is useful when we need to compute a probability for a sequence of
events that we can observe in the world. In many cases, however, the events we are
interested in may not be directly observable in the world. For example, in part-of-
speech tagging, we don’t observe part-of-speech tags in the world; we see words and have to infer the correct 
tags from the word sequence. Hence, we call the part-of-speech tags hidden because they are not observed.
"""
"""Imagine that you are a climatologist in the year 2799 studying the history of global
warming. You cannot find any records of the weather in Baltimore, Maryland, for
the summer of 2007, but you do find Jason Eisner’s diary, which lists how many ice
creams Jason ate every day that summer. Our goal is to use these observations to
estimate the temperature every day. We’ll simplify this weather task by assuming
there are only two kinds of days: cold (C) and hot (H). So the Eisner task is as
follows:
Given a sequence of observations O, each observation an integer cor-
responding to the number of ice creams eaten on a given day, figure
out the correct ‘hidden’ sequence Q of weather states (H or C) which
caused Jason to eat the ice cream."""
""""""


# In[30]:

"""A first-order hidden Markov model instantiates two simplifying assumptions.
First, as with a first-order Markov chain, the probability of a particular state depends
only on the previous state:
    Markov Assumption: P(qi |q1 ...qi−1 ) = P(qi |qi−1 )
Second, the probability of an output observation oi depends only on the state that
produced the observation qi and not on any other states or any other observations:
    Output Independence: P(oi |q1 . . . qi , . . . , qT , o1 , . . . , oi , . . . , oT ) = P(oi |qi )"""
""""""


# In[31]:

"""There is a (non-zero) probability of transitioning between any two states. Such an HMM is called 
a fully connected or ergodic HMM. Sometimes, however, we have HMMs in which many of the transitions 
between states have zero probability."""
""""""


# In[33]:

"""hidden Markov models should be characterized by three fundamental problems:
    Problem 1 (Likelihood): Given an HMM λ = (A, B) and an observation sequence O, determine the 
    likelihood P(O|λ ).
    Problem 2 (Decoding): Given an observation sequence O and an HMM λ =(A, B), discover the best hidden state 
    sequence Q.
    Problem 3 (Learning): Given an observation sequence O and the set of states in the HMM, learn the HMM 
    parameters A and B."""
""""""


# In[91]:

import numpy as np
import math
from math import log10
import sys


# In[117]:

#Likelihood Computation: The Forward Algorithm
#Our first problem is to compute the likelihood of a particular observation sequence say 3 1 3?
"""For a Markov chain, where the surface observations are the same as the hidden
events, we could compute the probability of 3 1 3 just by following the states labeled
3 1 3 and multiplying the probabilities along the arcs. For a hidden Markov model,
things are not so simple. We want to determine the probability of an ice-cream
observation sequence like 3 1 3, but we don’t know what the hidden state sequence
is!"""
#O: Observations, a:State Transition Prob, b:Emission Prob, pi: Initial Prob. 
#a[i][j]: Prob of transition from state i to state j
#b[i][j]: prob of emitting observation j at state i
#pi[i]: Initial Prob.of state i
#P(O|a,b,pi)=P(O|X=x1,a,b,pi)+P(O|X=x2,a,b,pi)+......+P(O|X=xn,a,b,pi) s.t. x1,x2,....,xn are all possible sequences
#alpha[t][i]:Partial Observation sequence upto time t is generated and we are state i
#alpha[t][i]= alpha[t-1][0]*a[0][i]*b[i][t]+
#    alpha[t-1][1]*a[1][i]*b[i][t]+....+alpha[t-1][n-1]*a[n-1][i]*b[i][t]#Assuming we have n states
def computeAlpha(observations,a,b,pi,alphaDP):
    statesC=a.shape[0]
    timePts=observations.shape[0]
    if timePts<1:
        return
    for i in np.arange(statesC):
        alphaDP[0][i]=-(log10(pi[i])+log10(b[i][observations[0]]))
    for t in np.arange(1,timePts):
        for i in np.arange(statesC):
            for j in np.arange(statesC):
                alphaDP[t][i]=alphaDP[t][i]-(log10(alphaDP[t-1][j])+log10(a[j][i])+log10(b[i][observations[t]]))
def observationProb(observations,a,b,pi,alphaDP):
    computeAlpha(observations,a,b,pi,alphaDP)
    print(alphaDP)
    timePts=observations.shape[0]
    stateC=a.shape[0]
    ans=0.0
    for i in np.arange(stateC):
        ans+=alphaDP[timePts-1][i]
    return ans


# In[118]:

observations=np.array([0,1,0,2])
A=np.array([[0.7,0.3],[0.4,0.6]])
B=np.array([[0.1,0.4,0.5],[0.7,0.2,0.1]])
pi=np.array([0.6,0.4])
alphaDP=np.zeros(shape=(observations.shape[0],A.shape[0]))# Count_of_Observations*Count_of_Hidden_States
observationProb(observations,A,B,pi,alphaDP)


# In[121]:

"""Problem 2 (Decoding): The Viterbi Algorithm -
Decoding: Given as input an HMM λ = (A, B, pi) and a sequence of observations O = o1 , o2 , ..., oT
find the most probable sequence of states Q = q1 q2 q3 . . . qT 
"""
"""Note that the Viterbi algorithm is identical to the forward algorithm except that it takes the max over the
previous path probabilities whereas the forward algorithm takes the sum. Note also
that the Viterbi algorithm has one component that the forward algorithm doesn’t have: backpointers. The reason 
is that while the forward algorithm needs to produce an observation likelihood, the Viterbi algorithm must 
produce a probability and also the most likely state sequence. We compute this best state sequence by keeping
track of the path of hidden states that led to each state, as suggested in Fig. 8.12, and then at the end backtracing 
the best path to the beginning (the Viterbi backtrace).
"""
def ViterbiAlgo(observations,a,b,pi,viterbiDP,viterbiBackPtr):
    statesC=a.shape[0]
    timePts=observations.shape[0]
    if statesC<1:
        return
    for i in np.arange(statesC):
        viterbiDP[0][i]=-(log10(pi[i])+log10(b[i][observations[0]]))
    for t in np.arange(1,timePts):
        for i in np.arange(statesC):
            maxSoFar=0
            maxJ=-1
            for j in np.arange(statesC):
                tempVal=-(log10(viterbiDP[t-1][j])+log10(a[j][i]))
                if tempVal>maxSoFar:
                    maxSoFar=tempVal
                    maxJ=j
            viterbiDP[t][i]=tempVal-log10(b[i][observations[t]])
            viterbiBackPtr[t][i]=maxJ
def getOptimalHiddenStates(observations,a,b,pi,viterbiDP,viterbiBackPtr):
    ViterbiAlgo(observations,a,b,pi,viterbiDP,viterbiBackPtr)
    print(viterbiDP)
    print(viterbiBackPtr)
    timePts=observations.shape[0]
    stateC=a.shape[0]
    ans=0.0
    endState=-1
    for i in np.arange(stateC):
        if ans<alphaDP[timePts-1][i]:
            ans=alphaDP[timePts-1][i]
            endState=i
    stateSeq=np.zeros(shape=(timePts))
    stateSeq[timePts-1]=endState
    i=timePts-2
    prevState=endState
    while i>=0:
        prevState=viterbiBackPtr[i+1][prevState]
        stateSeq[i]=prevState
        i=i-1
    return {"Probability":ans,"Sequence":stateSeq}


# In[122]:

observations=np.array([0,1,0,2])
A=np.array([[0.7,0.3],[0.4,0.6]])
B=np.array([[0.1,0.4,0.5],[0.7,0.2,0.1]])
pi=np.array([0.6,0.4])
viterbiDP=np.zeros(shape=(observations.shape[0],A.shape[0]))# Count_of_Observations*Count_of_Hidden_States
viterbiBackPtr=np.zeros(shape=(observations.shape[0],A.shape[0]))# Count_of_Observations*Count_of_Hidden_States
getOptimalHiddenStates(observations,A,B,pi,viterbiDP,viterbiBackPtr)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



