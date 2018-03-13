## Invert Brightfield images: background black/foreground white

invert <- function(x) {
    if(mean(x) > .3)
        x <- 1 - x
    x
}

## Global contrast normalization Goodfellow p. 442
gcn <- function(y, s = 1, lambda = 0.1, epsilon = 1e-8) {
    z <- y - mean(y)
    s*z/max(epsilon, sqrt(lambda + mean(z^2)))
}

to_gray_scale <- function(x) {
    y <- rgbImage(red = getFrame(x, 1),
                  green = getFrame(x, 2),
                  blue = getFrame(x, 3))
    y <- channel(y, mode="luminance")
    dim(y) <- c(dim(y), 1)
    y
}

preprocess_image <- function(file, shape = NULL){
    image <- readImage(file, type="png")[,,1:3]         ## drop fourth channel
    if(!is.null(shape))
        image <- resize(image, w = shape[1], h = shape[2])  ## make all images of dimensions
    image <- normalize(image)                           ## standardize between [0, 1]
    image <- invert(image)                              ## invert brightfield
    imageData(image)                                    ## return as array
}

preprocess_masks <- function(encoding, old_shape, new_shape = NULL){
    masks <- Image(rle2masks(encoding, old_shape))
    if(!is.null(new_shape))
        masks <- resize(masks, w = new_shape[1], h = new_shape[2])
    masks <- imageData(masks)
    dim(masks) <- c(dim(masks), 1) ##masks have no color channels
    masks
}

postprocess_image <- function(image, shape){
    image <- resize(image, w = shape[1], h = shape[2]) ## resize to origal dimensions
    image2rle(image)                                   ## encoding     
}
