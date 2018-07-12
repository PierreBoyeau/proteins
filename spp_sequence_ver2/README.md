### Outline ###

* This program extracts characteristic patterns for labels from sequence data.
* Please refer to the following papers for details of the algorithm.
 - Nakagawa, Kazuya, et al. "Safe pattern pruning: An efficient approach for predictive pattern mining." KDD2016
 - Sakuma, Takuto, et al. "Efficient Learning Algorithm for Sparse SubSequence Pattern-based Classification and Applications to Comparative Animal Trajectory Data Analysis"


### Can do ###

* Pattern extraction that allowed jumping
* Pattern extraction using frequency
* Restriction on the length of extracted patterns
* Parameter selection by Cross Validation
* Elimination of completely overlapping patterns by CloSpan

### Setup ###

* make all

### How to use ###
train [-options] input_file

* options:  
    -u : problem type (default 1)  
    　　1 -- regularization path computation for L1-reguralized L2-SVM
    　　2 -- regularization path computation for Lasso    
    -t : learning lambda index(when do not cv) (default:most minimum lambda)  
    -m : minimum supportSum (default 1)  
    -L : maximum length of pattern (default 10)  
    -T : the number of regularization parameter (default 100)  
    -r : lambda min ratio (default 2.0)  
    -i : max outer iteration in optimization (default 1000000)  
    -f : frequency of calculate duality gap and convergence check (default 50)  
    -e : convergence criterion of duality gap (default 1e-6)  
    -F : name of reslut file (default output/result.csv)  
    -p : maximum interval of event (default 0|-1:none)  
    -c : whether to do cross validation (default 0:do not|1:do)  
    -k : k-fold of k (when do cross validation)(default:10)  
    -a : times to do cross validation(default:10)  
    -C : whether to do CloSpan (default 0:do not|1:do)  
    -M : whether to do Multiprocess  (default 0:do not|1:do)  
    -P : whether to do cross validation for Performance evaluation (default 0:do not|1:do)  
    -s : the Mode of counting supportSum(default 0),0 is 0 or 1 per record,1 is the number of pattern  


### Running example ###

simple
* ./train -L 50 -F output/hoge_result.csv data/hoge.txt

Takuto often use
* ./train -L 100 -p 0 -s 0 -c 1 -C 0 -M 1 -F output/hoge_result.csv data/hoge.txt
