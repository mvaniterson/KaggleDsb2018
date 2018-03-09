
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
    y_true_f <- k_flatten(y_true)
    y_pred_f <- k_flatten(y_pred)
    intersection <- k_sum(y_true_f * y_pred_f)
    (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
}
attr(dice_coef, "py_function_name") <- "dice_coef"


dice_coef_loss <- function(y_true, y_pred) -dice_coef(y_true, y_pred)
attr(dice_coef_loss, "py_function_name") <- "dice_coef_loss"


optimize_intersections <- function(y_pred, y_true, verbose = FALSE) {

    cm <- table(y_pred, y_true, exclude = 0)

    if(verbose)
        message("trace before: ", sum(diag(cm)))

    labels <- rownames(cm)
    for(j in 1:min(nrow(cm), ncol(cm))) {
        i <- which.max(cm[,j])
        if(i != j & cm[i,j] != 0) {
            ##swap
            tmp <- cm[j,j]
            cm[j,j] <- cm[i,j]
            cm[i,j] <- tmp
            tmp <- labels[j]
            labels[j] <- labels[i]
            labels[i] <- tmp
        }
    }

    if(verbose)
        message("trace after: ", sum(diag(cm)))

    y_prd <- factor(y_pred)
    levels(y_prd) <- c("0", labels)
    as.integer(as.character(y_prd))
}


iou <- function(y_true, y_pred, optimize = TRUE, verbose = FALSE){

    if(verbose)
        message("Calculate IoU")

    y_true <- as.vector(y_true)
    y_pred <- as.vector(y_pred)
    nbins <- max(y_true, y_pred)

    tabulate <- function(x, nbins) {
        sizes <- numeric(nbins)
        for(k in 1:nbins)
            sizes[k] <- sum(x == k)
        sizes
    }

    ##find relabeling of y_pred that maximise the intersections
    if(optimize)
        y_pred <- optimize_intersections(y_pred, y_true, verbose)

    intersections <- tabulate(sqrt(y_true*y_pred), nbins = nbins)

    true_sizes <- tabulate(y_true, nbins = nbins)
    pred_sizes <- tabulate(y_pred, nbins = nbins)
    unions <- true_sizes + pred_sizes - intersections

    ##IoU
    intersections/unions
}

mean_precision <- function(y_true, y_pred, thresholds = seq(0.5, 0.95, 0.05), optimize = FALSE, verbose = FALSE) {

    if(verbose)
        message("Calculate mean precision")

    ious <- iou(y_true, y_pred, optimize, verbose)

    ground_truth <- logical(max(y_true, y_pred))
    ground_truth[1:max(y_true)] <- TRUE

    precision <- 0
    for(t in thresholds) {
        tp <- sum(ious[ground_truth] > t)
        fp <- sum(ious[!ground_truth] > t)
        fn <- sum(ious[ground_truth] <= t)
        precision <- precision + tp/(tp+fp+fn)
    }
    precision/length(thresholds)
}
