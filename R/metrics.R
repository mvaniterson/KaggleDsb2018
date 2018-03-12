
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
    y_true_f <- k_flatten(y_true > .5)
    y_pred_f <- k_flatten(y_pred > .5)
    intersection <- k_sum(y_true_f * y_pred_f)
    (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
}
attr(dice_coef, "py_function_name") <- "dice_coef"

dice_coef_loss <- function(y_true, y_pred) -dice_coef(y_true, y_pred)
attr(dice_coef_loss, "py_function_name") <- "dice_coef_loss"

jaccard_coef <- function(y_true, y_pred) {
    y_true_f <- k_flatten(y_true > .5)
    y_pred_f <- k_flatten(y_pred > .5)
    intersection <- k_sum(y_true_f * y_pred_f)
    intersection / (k_sum(y_true_f) + k_sum(y_pred_f) - intersection)
}
attr(jaccard_coef, "py_function_name") <- "jaccard_coef"

jaccard_coef_loss <- function(y_true, y_pred) -jaccard_coef(y_true, y_pred)
attr(jaccard_coef_loss, "py_function_name") <- "jaccard_coef_loss"


iou <- function(y_true, y_pred){

    y_true <- as.vector(y_true)
    y_pred <- as.vector(y_pred)
    nbins <- max(y_true, y_pred)
    intersections <- tabulate(sqrt(y_true*y_pred), nbins = nbins)

    true_sizes <- tabulate(y_true, nbins = nbins)
    pred_sizes <- tabulate(y_pred, nbins = nbins)
    unions <- true_sizes + pred_sizes - intersections

    intersections/unions
}

mean_precision <- function(y_true, y_pred, thresholds = seq(0.5, 0.95, 0.05)) {

    ious <- iou(y_true, y_pred)

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
