## list2tensor i.e., inverse of purrr's array_branch
list2tensor <- function(xList, index) {
    xTensor <- simplify2array(xList)
    indices <- 1:length(dim(xTensor))
    aperm(xTensor, c(index, indices[-index]))    
}
