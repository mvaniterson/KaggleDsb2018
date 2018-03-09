
## ----requirements--------------------------------------------------------
library(keras)
library(tidyverse)
library(EBImage)

src_files <- list.files("R", pattern="*.R", full.names = TRUE)
tmp <- sapply(src_files, source, .GlobalEnv)


options(EBImage.display = "raster")
set.seed(12345)

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
HEIGHT = 256/4
WIDTH = 256/4
CHANNELS = 1
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
BATCH_SIZE = 16
EPOCHS = 10

## ----prepare data------------------------------------------------------------
train_data <- read_csv("../input/stage1_train_labels.csv") %>%
    group_by(ImageId) %>%
    summarize(EncodedPixels = list(EncodedPixels)) %>%
    mutate(ImageFile = file.path(TRAIN_PATH, ImageId, "images", paste0(ImageId, ".png")),
           MaskPath = file.path(TRAIN_PATH, ImageId, "masks"),
           MaskFiles = map(MaskPath, list.files, pattern="*.png", full.names = TRUE),
           ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))

test_data <- tibble(ImageId = dir(TEST_PATH)) %>%
    mutate(ImageFile = file.path(TEST_PATH, ImageId, "images", paste0(ImageId, ".png")),
           ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))

train_data %>%
    glimpse()

test_data %>%
    glimpse()

## ----Display some images----------------------------------------------------------
input_batch <- sample_n(train_data, 32) %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

input_batch

X <- simplify2array(input_batch$X)
X <- aperm(X, c(4, 1, 2, 3))

Y <- simplify2array(input_batch$Y)
Y <- aperm(Y, c(4, 1, 2, 3))

display(combine(Y[1,,,], X[1,,,]), all = TRUE)
display(combine(Y[5,,,], X[5,,,]), all = TRUE)
display(combine(Y[32,,,], X[32,,,]), all = TRUE)


## ----Define model----------------------------------------------------------

model <- unet(shape = SHAPE, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.3))

##model <- unet(shape = SHAPE, nlevels = 4, filters = 16, dropouts = c(0.1, 0.1, 0.2, 0.2, 0.3))

model <- model %>%
    compile(
        optimizer = 'adam',
        loss = dice_coef_loss,
        metrics = c(dice_coef)
    )

summary(model)

## ----fit-----------------------------------------------------------------

input <- train_data %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

X <- simplify2array(input$X)
X <- aperm(X, c(4, 1, 2, 3))

Y <- simplify2array(input$Y)
Y <- aperm(Y, c(4, 1, 2, 3))

history <- model %>%
    fit(X, Y,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_split = 0.2)

## ----inspect model--------------------------------------------------------
plot(history)

save_model_hdf5(model, filepath="unet_model16_256.hdf5")

##model <- load_model_hdf5("unet_model.hdf5", custom_objects=c(dice_coef_loss=dice_coef_loss, dice_coef=dice_coef))

## ----evaluate model-------------------------------------------------------

input_batch <- sample_n(train_data, 32) %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

input_batch

X <- simplify2array(input_batch$X)
X <- aperm(X, c(4, 1, 2, 3))

Y <- simplify2array(input_batch$Y)
Y <- aperm(Y, c(4, 1, 2, 3))

display(combine(Y[1,,,], X[1,,,]), all = TRUE)
display(combine(Y[5,,,], X[5,,,]), all = TRUE)
display(combine(Y[32,,,], X[32,,,]), all = TRUE)

Y_hat <- predict(model, x = X)

display(combine(Y[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(Y[5,,,], Y_hat[5,,,]), all = TRUE)
display(combine(Y[32,,,], Y_hat[32,,,]), all = TRUE)

##convert to binary images/masks
Y_hat <- Y_hat > 0.5
Y <- Y > 0.5

masks_pred <- array_branch(Y_hat, 1) %>%
    map(bwlabel)

masks <- array_branch(Y, 1) %>%
    map(bwlabel)

display(colorLabels(combine(masks[[1]], masks_pred[[1]])), all = TRUE)
display(colorLabels(combine(masks[[5]], masks_pred[[5]])), all = TRUE)
display(colorLabels(combine(masks[[32]], masks_pred[[32]])), all = TRUE)

##Estimate mean precision
mp_unet_opt <- map2_dbl(masks, masks_pred, mean_precision, optimize = TRUE)
mp_unet <- map2_dbl(masks, masks_pred, mean_precision, optimize = FALSE)
mean(mp_unet_opt)
mean(mp_unet)
boxplot(cbind(mp_unet, mp_unet_opt))

## ----predict test data----------------------------------------------------------

test_data <- test_data %>%
    mutate(X = map(ImageFile, preprocess_image, shape = SHAPE))

test_data

X <- simplify2array(test_data$X)
X <- aperm(X, c(4, 1, 2, 3))

display(X[1,,,], all = TRUE)
display(X[5,,,], all = TRUE)
display(X[32,,,], all = TRUE)

Y_hat <- predict(model, x = X)

display(combine(X[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(X[5,,,], Y_hat[5,,,]), all = TRUE)
display(combine(X[32,,,], Y_hat[32,,,]), all = TRUE)

## ----submission----------------------------------------------------------

postprocess_image <- function(image, shape){
    image <- resize(image[,,1], w = shape[1], h = shape[2])
    image <- bwlabel(image)
    image2rle(image)
}

submission <- test_data %>%
    add_column(Masks = array_branch(Y_hat, 1)) %>%
    mutate(EncodedPixels = map2(Masks, ImageShape, postprocess_image))

submission

rsamples <- sample_n(submission, 3) %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

rsamples

X <- simplify2array(rsamples$X)
X <- aperm(X, c(4, 1, 2, 3))

Y <- simplify2array(rsamples$Y)
Y <- aperm(Y, c(4, 1, 2, 3))

display(combine(Y[1,,,], X[1,,,]), all = TRUE)
display(combine(Y[2,,,], X[2,,,]), all = TRUE)
display(combine(Y[3,,,], X[3,,,]), all = TRUE)

submission <- submission %>%
    unnest(EncodedPixels) %>%
    mutate(EncodedPixels = as.character(EncodedPixels)) %>%
    select(ImageId, EncodedPixels)

submission

write_csv(submission, "submission.csv")




