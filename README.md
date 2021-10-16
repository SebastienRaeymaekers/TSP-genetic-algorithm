<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Traveling salesman problem using a genetic algorithm</h3>

  <p align="center">
    Given a distance matrix between n cities, find the shortest path that goes through each city exactly once using a genetic algorithm.
    <br />
    <br />
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## About The Project

As input a distance matrix in .csv-file format is required. Also, a path representation has to be chosen.
A permutation of the cities numbers is chosen for this, which means a path will be represented by an 
array of numbers in which the path will follow the city with the corresponding number from left to right.
Example: path `[4,2,1,3]` means the path will first go through city number 4, then city number 2 etc.

The genetic algorithm is run using the following steps:
* First, an initial population has to be chosen. This happens using a KNN approach where cities are grouped in
clusters as much as possible by putting cities from the same cluster next to each other in the path representation.
To keep the cluster sizes relatively small, `abs(sqrt(N))` cluster sizes are determined by dividing the total number of cities over the clusters, while ensuring
an equal spread and with N the total number of cities.
* Secondly, a k-tournament selection procedure selects the parents for reproduction of the next generation. 
Overall it performs very well as it’s easy to compute because no sorting is needed, k will be relatively 
low with respect to the population size, it can preserve individuals with bad fitnesses which is 
good for exploration and at last, the selection pressure can easily be increased/decreased by 
decreasing/increasing k.
* Then, inversion mutation is used. This operator promotes exploitation and there is a degree of 
randomness, since the slice that is inverted in an individual is selected randomly by the algorithm.
The rate of mutation is controlled by a predefined, fixed parameter. Although some more randomness
is added by using a scramble mutation in one of the three islands (see island explanation below).
* After this, the order crossover recombination operator is executed. This one was chosen because it combines and maintains the
relative order of cities, which is what is needed as the cycle notation is used. There are however some drawbacks
to this method. The sub-tour could be very large, thus not making good use of the second individuals cities.
By having to skip values to ensure no overlap, parts of the second individual are not conserved optimally.
* Then, as elimination operator we have the round-robin tournament scheme with q equal to 8 that is 
used instead of a k-tournament selection. In a round-robin tournament, the chance of individuals 
with relatively bad fitnesses getting chosen is a little higher than with a k-tournament selection scheme. 

Mechanisms used to improve diversity: An island model is implemented. The island model runs 
different evolutionary algorithms in parallel. To enhance diversity, the islands themselves need 
to be diverse from one another. This is why it is possible to pass different functions for the initialisation 
method, selection method, etc. and pass different values for the crossover rate, mutation rate etc. 
for the evolutionary algorithm as parameters when creating new T sp-objects. Alongside the 
different characteristics, the islands should also have some form of interaction. Every 50th 
iteration each island will perform the recombination step with another randomly chosen island, 
while keeping the best individual of each of the two chosen islands intact.

Mechanisms used to improve local search: as mentioned before, during the initialisation, a 
KNN-clustering algorithm is run. It is clear that cities and their closest neighbours, should be 
close to each other in the numpy array representation because of the cycle notation used.

Stopping criterion: a simple iteration counter is kept and the algorithm would stop after 
10000 iterations or when the time limit of 5 minutes is surpassed.

Parameter selection: self-adaptivity is added. Finding optimal parameters is not easy, yet an 
overlooked aspect of evolutionary algorithms. That’s why self-adaptivity is used. This way the
algorithm itself can see which parameters give the best solutions.

Output: An approximation of the shortest distance between all cities.


### Built With
* [Python](https://www.python.org/)
* [Numpy](https://numpy.org/)


<!-- GETTING STARTED -->
## Getting Started

Simply call the tsp.py script with the name of a csv file that is the distance matrix as an argument.

<!-- LICENSE -->
## License

Distributed under the MIT License.

