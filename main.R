
## ----requirements--------------------------------------------------------
library(keras)
library(tidyverse)
library(EBImage)

src_files <- list.files("R", pattern="*.R", full.names = TRUE)
tmp <- sapply(src_files, source, .GlobalEnv)

options(EBImage.display = "raster")

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
HEIGHT = 256/2
WIDTH = 256/2
CHANNELS = 3
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
BATCH_SIZE = 16
EPOCHS = 25

## ----prepare data------------------------------------------------------------
train_data <- read_csv("../input/stage1_train_labels.csv") %>%
    group_by(ImageId) %>%
    summarize(EncodedPixels = list(EncodedPixels)) %>%
    mutate(ImageFile = file.path(TRAIN_PATH, ImageId, "images", paste0(ImageId, ".png")),
           MaskPath = file.path(TRAIN_PATH, ImageId, "masks"),
           MaskFiles = map(MaskPath, list.files, pattern="*.png", full.names = TRUE),
           ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))

train_data %>%
    glimpse()

## ----Display some images----------------------------------------------------------
input_batch <- sample_n(train_data, 3) %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

input_batch

display(combine(input_batch$Y[[1]], input_batch$X[[1]]), all = TRUE)
display(combine(input_batch$Y[[2]], input_batch$X[[2]]), all = TRUE)
display(combine(input_batch$Y[[3]], input_batch$X[[3]]), all = TRUE)

## ----Define model----------------------------------------------------------

model <- unet(shape = SHAPE, nlevels = 2, nfilters = 16, dropouts = c(0.1, 0.1, 0.2))

##model <- unet(shape = SHAPE, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.3))

##model <- unet(shape = SHAPE, nlevels = 4, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.2, 0.3))

model <- model %>%
    compile(
        optimizer = 'adam',
        loss = jaccard_coef_loss,
        metrics = c(jaccard_coef)
    )

model <- model %>%
    compile(
        optimizer = 'adam',
        loss = dice_coef_loss,
        metrics = c(dice_coef)
    )

summary(model)

## ----fit-----------------------------------------------------------------

input <- sample_n(train_data, nrow(train_data)) %>%
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

save_model_hdf5(model, filepath="unet_model_2_16_128.hdf5")

##model <- load_model_hdf5("unet_model.hdf5", custom_objects=c(dice_coef_loss=dice_coef_loss, dice_coef=dice_coef))

## ----evaluate model-------------------------------------------------------

Y_hat <- predict(model, x = X)

display(combine(Y[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(Y[100,,,], Y_hat[100,,,]), all = TRUE)
display(combine(Y[320,,,], Y_hat[320,,,]), all = TRUE)


##convert to binary images/masks
Y_hat <- Y_hat > 0.5
Y <- Y > 0.5

masks_pred <- array_branch(Y_hat, 1) %>%
    map(bwlabel)

masks <- array_branch(Y, 1) %>%
    map(bwlabel)

display(colorLabels(combine(masks[[10]], masks_pred[[10]])), all = TRUE)
display(colorLabels(combine(masks[[500]], masks_pred[[500]])), all = TRUE)
display(colorLabels(combine(masks[[632]], masks_pred[[632]])), all = TRUE)

##Estimate mean precision
mp <- map2_dbl(masks, masks_pred, mean_precision)
mean(mp)
hist(mp, n=100)

## ----predict test data----------------------------------------------------------

test_data <- tibble(ImageId = dir(TEST_PATH)) %>%
    mutate(ImageFile = file.path(TEST_PATH, ImageId, "images", paste0(ImageId, ".png")),
           ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]))

test_data %>%
    glimpse()

X <- test_data %>%
    mutate(X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    pull(X)

X <- simplify2array(X)
X <- aperm(X, c(4, 1, 2, 3))

Y_hat <- predict(model, x = X)

display(combine(X[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(X[5,,,], Y_hat[5,,,]), all = TRUE)
display(combine(X[32,,,], Y_hat[32,,,]), all = TRUE)

## ----submission----------------------------------------------------------

submission <- test_data %>%
    add_column(Masks = array_branch(Y_hat, 1)) %>%
    mutate(EncodedPixels = map2(Masks, ImageShape, postprocess_image))

## check encoding
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

## check rle encoding
rsamples <- sample_n(train_data, 5)

im <- readImage(rsamples$MaskFiles[[1]])

##label
for(i in 1:numberOfFrames(im))
    im[,,i] <- i*im[,,i]
im <- Reduce("+", getFrames(im))

dim(im)
range(im)

image(im)

image(t(im)[1:200, 200:520])

bw <- bwlabel(im>0)
image(bw)

im <- readImage(rsamples$MaskFiles[[1]])
im <- Reduce("+", getFrames(im))

