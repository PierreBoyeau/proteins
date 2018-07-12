
## Folders

- data
- obj
- output
- paper

## Parameters

- Related to lambda: r: corresponds to how smaller lambda_min is compared to lambda_max
    T: number of points in grid; a : number of cross validations for lambda selection


an option is a triplet { aK, aAve, aLossType} where aAve is the number of CV done for lambda selection

## Functions

- train.cc

    2 types of CV can be performed:
    1) For performance
    2) For lambda selection : decreasing with warm starts each time
    The code is divided in 2 parts, depending if we do performance CV or not.



- crossvalidation.h
- database.h
    
    Class that deal with opening input file.
    File are read the following way:
    - First delimiter is expected to be the label, as a float/int
    - All words that appear after that are expected to be ints, and encode the graph.
    
    Question : What about sets?

- modelselector.h
- prefixspan.h
    Seems very important.
    Initialized using features and labels.
    
    Not sure I have completely understood the interval criteria : l.208

- learnerspp.h
    Virtual class used by lasso/svm
- lassospp.h
    Used by default, for REGRESSION TASKS!
- svmspp.h
    Used by default, for CLASSIFICATION!