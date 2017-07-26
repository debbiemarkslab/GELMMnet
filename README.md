
# GELMMnet - Generalized network-based elastic-net linear mixed model

Implementation of a generalized elastic net LMM for (mainly) GWAS analysis. 
It also implements the exact post selection inference method proposed by Lee et al. to generate
accurate confidence interval and p-values adjusted for elastic net.


The implementation is based on Barbara Rakitsch's implementation of LMM-Lasso (https://github.com/BorgwardtLab/LMM-Lasso)
, Artem Skolov's implementation of GELnet (https://github.com/cran/gelnet), 
and selectiveInference by the Selective Inference Team (https://github.com/selective-inference/Python-software/).

For multiprocessing we are using Pathos.
(https://github.com/uqfoundation/pathos)

The software is released under the GNU General Public License.

**Author:** 

Benjamin Schubert    
Debora Marks and Chris Sander Group    
Systems and Cell Biology,   
Harvard Medical School,   
200 Longwood Avenue, Boston, 02115 MA, USA      


**References:**

1) Rakitsch, B., Lippert, C., Stegle, O., & Borgwardt, K. (2012). 
A Lasso multi-marker mixed model for association mapping with 
population structure correction. Bioinformatics, 29(2), 206-214.

2) Sokolov, A., Carlin, D. E., Paull, E. O., Baertsch, R., & Stuart, J. M. (2016). 
Pathway-based genomics prediction using generalized elastic net. 
PLoS Computational Biology, 12(3), e1004790.

3) Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016). 
Exact post-selection inference, with application to the lasso. 
The Annals of Statistics, 44(3), 907-927. Chicago	


4) McKerns, M., & Aivazis, M. pathos: a framework for heterogeneous computing, 2010.
Chicago	

