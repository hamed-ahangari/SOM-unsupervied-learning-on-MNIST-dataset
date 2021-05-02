
# SOM unsupervised learning on MNIST
In this repository, I am publishing codes and results of one of the assignments in the "Neural Network and Deep Learning" course, which I have passed during my Master's degree at the University of Tehran.
The assignment's goal was to train the Self-Organizing Map algorithm on the MNIST dataset.

**Table of Contents**
- [SOM unsupervised learning on MNIST](#som-unsupervised-learning-on-mnist)
  - [Self Organising Maps – Kohonen Maps](#self-organising-maps--kohonen-maps)
  - [What really happens in SOM?](#what-really-happens-in-som)
    - [SOM algorithm](#som-algorithm)
  - [The code, variations, and results](#the-code-variations-and-results)
    - [Linear topology, Random weights](#linear-topology-random-weights)
    - [Linear topology, Sampled weights](#linear-topology-sampled-weights)
    - [Grid topology, Random weights](#grid-topology-random-weights)
    - [Grid topology, Sampled weights](#grid-topology-sampled-weights)
  - [Other references](#other-references)
  - [Further reading](#further-reading)

## Self Organising Maps – Kohonen Maps
The  [Self-Organizing Map](https://en.wikipedia.org/wiki/Self-organizing_map "Wikipedia")  (SOM) is an unsupervised learning algorithm introduced by Kohonen [[1]](https://scholar.google.com/scholar_lookup?title=Self-Organizing%20Maps&publication_year=1995&author=T.%20Kohonen "Google Scholar"). In the area of  [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network "Wikipedia"), the SOM is an excellent data-exploring tool as well [[2]](https://scholar.google.com/scholar?q=A%20new%20approach%20for%20data%20clustering%20%20visualization%20using%20self-organizing%20maps "Google Scholar"). It can project high-dimensional patterns onto a low-dimensional topology map. The SOM map consists of a one or two dimensional (2-D) grid of nodes. These nodes are also called neurons. Each neuron's weight vector has the same dimension as the input vector. The SOM obtains a statistical feature of the input data and is applied to a wide field of data classification [[3]](https://scholar.google.com/scholar?q=Y.%20Cheng,%20Clustering%20with%20competing%20self-organizing%20maps,%20in:%20Proc.%20of%20IJCNN,%20vol.%20IV,%20pp.%20785790,%201992. "Clustering with competing self-organizing maps"), [[4]](https://scholar.google.com/scholar?q=W.%20Wan,%20D.%20Fraser,%20M2dSOMAP:%20clustering%20and%20classification%20of%20remotely%20sensed%20imagery%20by%20combining%20multiple%20Kohonen%20self-organizing%20maps%20and%20associative%20memory,%20in:%20Proc.%20of%20IJCNN,%20vol.%20III,%20pp.%2024642467,%201993. "M2dSOMAP: clustering and classification of remotely sensed imagery by combining multiple Kohonen self-organizing maps and associative memory"), [[5]](https://scholar.google.com/scholar_lookup?title=Clustering%20of%20the%20self-organizing%20map&publication_year=2002&author=J.%20Vesanto&author=E.%20Alhoniemi "Clustering of the self-organizing map"), [[6]](https://scholar.google.com/scholar_lookup?title=Unsupervised%20speaker%20recognition%20based%20on%20competition%20between%20self-organizing%20maps&publication_year=2002&author=Lapidot&author=H.%20Guterman&author=A.%20Cohen "Unsupervised speaker recognition based on competition between self-organizing maps"). SOM is based on competitive learning. In competitive learning [[7]](https://scholar.google.com/scholar_lookup?title=Feature%20discovery%20by%20competitive%20learning&publication_year=1985&author=E.%20Rumelhart&author=D.%20Zipser "Feature discovery by competitive learning"), neuron activation is a function of the distance between neuron weight and input data. An activated neuron learns the most, and its weights are thus modified.

## What really happens in SOM?
Each data point in the dataset recognizes itself by competing for representation. SOM mapping steps start from initializing the weight vectors. From there, a sample vector is selected randomly, and the map of weight vectors is searched to find which weight best represents that sample. Each weight vector has neighboring weights that are close to it. The weight that is chosen is rewarded by being able to become more like that randomly selected sample vector. The neighbors of that weight are also rewarded by being able to become more like the chosen sample vector. This allows the map to grow and form different shapes. Most generally, they form square/rectangular/hexagonal/L shapes in 2D feature space.


<p align="center"><a href="https://commons.wikimedia.org/wiki/File:TrainSOM.gif#/media/File:TrainSOM.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/3/35/TrainSOM.gif" alt="TrainSOM.gif" height="300" width="300"></a><br>Training process of SOM on a two-dimensional data set<br>By <a href="//commons.wikimedia.org/w/index.php?title=User:Chompinha&amp;amp;action=edit&amp;amp;redlink=1" class="new" title="User:Chompinha (page does not exist)">Chompinha</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=77822988">Link</a></p>

### SOM algorithm
1.  Each node's weights are initialized.
2.  A vector is chosen at random from the set of training data.
3.  Every node is examined to calculate which one's weights are most like the input vector. The winning node is commonly known as the  **Best Matching Unit**  (BMU).
4.  Then, the neighborhood of the BMU is calculated. The amount of neighbors decreases over time.
5.  The winning weight is rewarded with becoming more like the sample vector. The neighbors also become more like the sample vector. The closer a node is to the BMU, the more its weights get altered, and the farther away the neighbor is from the BMU, the less it learns.
6.  Repeat step 2 for N iterations.

**Best Matching Unit**  is a technique that calculates the distance from each weight to the sample vector by running through all weight vectors. The weight with the shortest distance is the winner. There are numerous ways to determine the distance. However, the most commonly used method is the  [_Euclidean Distance_](https://en.wikipedia.org/wiki/Euclidean_distance "Wikipedia"), and that is what is used in the following implementation.

## The code, variations, and results
Based on connection topology and weight initialization, I have implemented the SOM clustering on the MNIST dataset in four different ways:
- Linear topology, Random weights initialization
- Linear topology, Sampled weights from data points
- Grid topology, Random weight initialization
- Grid topology, Sampled weights from data points

Codes of each implementation are placed in the "[Codes](https://github.com/hamed-ahangari/SOM-unsupervied-learning-on-MNIST-dataset/tree/main/Codes)" folder. Also, below you can see the animated gif of neurons' weights in each epoch during the training.
**Note:** weights of winner neurons at each epoch are illustrated with non-grey colormap (plasma).
### Linear topology, Random weights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/16l5Vd8_q007Ygupj3z929hOWwXKQdL7Z/view?usp=sharing)
Weights are initialized with random samples from a uniform distribution over `[0,  1)`.
![Linear topology, Random weight initialization](https://github.com/hamed-ahangari/SOM-unsupervied-learning-on-MNIST-dataset/raw/main/Images/LINEAR-topology_RANDOM-weights/gif/LINEAR-topology_RANDOM-weights.gif)


### Linear topology, Sampled weights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1D-SulHT6e2yOLkjRhW72L_41WbeMxGzA/view?usp=sharing)
Weights are initialized with samples from images in the dataset.
![Linear topology, Sampled weights](https://github.com/hamed-ahangari/SOM-unsupervied-learning-on-MNIST-dataset/raw/main/Images/LINEAR-topology_SAMPLED-weights/gif/LINEAR-topology_SAMPLED-weights.gif)

### Grid topology, Random weights

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/14dS8BII5YVH0jdyfrNe7AI93Et1QB9oe/view?usp=sharing)

Weights are initialized with random samples from a uniform distribution over `[0,  1)`.
![Grid topology, Random weights](https://github.com/hamed-ahangari/SOM-unsupervied-learning-on-MNIST-dataset/raw/main/Images/GRID-topology_RANDOM-weights/gif/GRID-topology_RANDOM-weights.gif)

### Grid topology, Sampled weights
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1k8ZxfqPWghaQd1GekfML0OMsOQaCIHRY/view?usp=sharing)
Weights are initialized with samples from images in the dataset.
![Grid topology, Sampled weights](https://github.com/hamed-ahangari/SOM-unsupervied-learning-on-MNIST-dataset/raw/main/Images/GRID-topology_SAMPLED-weights/gif/GRID-topology_SAMPLED-weights.gif)

## Other references
- [Chaudhary, Vikas, R. S. Bhatia, and Anil K. Ahlawat. "A novel Self-Organizing Map (SOM) learning algorithm with nearest and farthest neurons." _Alexandria Engineering Journal_ 53.4 (2014): 827-831.](https://www.sciencedirect.com/science/article/pii/S1110016814000970) (paper)
- [Applications of the growing self-organizing map, Th. Villmann, H.-U. Bauer, May 1998](https://www.sciencedirect.com/science/article/abs/pii/S092523129800037X "ScienceDirect") (paper)
- [Self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) (webpage)
- [Discovering SOM, an Unsupervised Neural Network](https://medium.com/neuronio/discovering-som-an-unsupervised-neural-network-12e787f38f9) (webpage)
- [Credit Card Fraud Detection using Self Organizing FeatureMaps](https://towardsdatascience.com/credit-card-fraud-detection-using-self-organizing-featuremaps-f6e8bca707bd) (webpage)
- [MiniSom](https://github.com/JustGlowing/minisom) (GitHub repository)
## Further reading
-   [Self Organizing Maps on the Glowing Python](https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html)
-   [Lecture notes from the Machine Learning course at the University of Lisbon](http://aa.ssdi.di.fct.unl.pt/files/AA-16_notes.pdf)
-   [Introduction to Self-Organizing](https://heartbeat.fritz.ai/introduction-to-self-organizing-maps-soms-98e88b568f5d)  by Derrick Mwiti
-   Video tutorial  [Self Organizing Maps: Introduction](https://www.youtube.com/watch?v=0qtvb_Nx2tA)  by SuperDataScience
-   [MATLAB Implementations and Applications of the Self-Organizing Map](http://docs.unigrafia.fi/publications/kohonen_teuvo/)  by Teuvo Kohonen (Inventor of SOM)
