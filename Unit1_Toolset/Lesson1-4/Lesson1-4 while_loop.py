"""
Use a while loop to solve the following problem: 
A slow, but determined, walker sets off from Leicester to cover the 102 miles to London at 2 miles per hour. 
Another walker sets off from London heading to Leicester going at 1 mile per hour. 
Where do they meet?
"""
>>> x=0
>>> walking = True
>>> while walking:
...     if 2*x + x !=102:
...             print("{} miles covered.".format(2*x + x))
...             x+=1
...             print("{} hours passed...".format(x))
...     else:
...             walking = False
... else:
...     print("they meet after {} hours' walking".format(x))
...