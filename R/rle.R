## Convert rle encoded masks to single images
rle2masks <- function(encodings, shape) {

    ## Convert rle encoded mask to image
    rle2mask <- function(encoding, shape){

        splitted <- as.integer(str_split(encoding, pattern = "\\s+", simplify=TRUE))
        positions <- splitted[seq(1, length(splitted), 2)]
        lengths <- splitted[seq(2, length(splitted), 2)] - 1

        ## decode
        mask_indices <- unlist(map2(positions, lengths, function(pos, len) seq.int(pos, pos+len)))

        if(max(mask_indices) > prod(shape)){
            print(max(mask_indices))
            print(prod(shape))
            stop("Encoding doesn't match with given shape!")
        }

        ## shape as 2D image
        mask <- numeric(prod(shape))
        mask[mask_indices] <- 1
        mask <- matrix(mask, nrow=shape[1], ncol=shape[2], byrow=TRUE)
        mask
    }

    masks <- matrix(0, nrow=shape[1], ncol=shape[2])
    for(i in 1:length(encodings))
        masks <- masks + rle2mask(encodings[i], shape)

    if(any(masks > 1))
        message("Overlapping masks!")
    masks
}

## Convert segmented image to rle encoded strings
image2rle <- function(image){

    labels <- 1:max(image) ## assuming background  == 0

    x <- as.vector(t(image)) ##colum-wise

    encoding <- rle(x)

    ## Adding start positions
    encoding$positions <- 1 + c(0, cumsum(encoding$lengths[-length(encoding$lengths)]))

    mask2rle <- function(label, enc) {
        indices <- enc$values == label
        list(position = enc$positions[indices][1],
             encoding = paste(enc$positions[indices], enc$lengths[indices], collapse=" "))
    }

    ##return encodings with increasing positions
    map_df(labels, mask2rle, encoding) %>%
        arrange(position) %>%
        pull(encoding)

}
