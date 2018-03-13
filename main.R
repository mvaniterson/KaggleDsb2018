
## ----requirements--------------------------------------------------------
library(keras)
library(tidyverse)
library(EBImage)

src_files <- list.files("R", pattern="*.R", full.names = TRUE)
tmp <- sapply(src_files, source, .GlobalEnv)

options(EBImage.display = "raster")

TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
BATCH_SIZE = 16
EPOCHS = 50

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

##model <- unet(shape = SHAPE, nlevels = 2, nfilters = 16, dropouts = c(0.1, 0.1, 0.2))
##model <- unet(shape = SHAPE, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.3))
model <- unet(shape = SHAPE, nlevels = 4, nfilters = 16, dropouts = c(0.1, 0.1, 0.2, 0.2, 0.3))

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

X <- list2tensor(input$X, 4)
Y <- list2tensor(input$Y, 4)
dim(X)


checkpoint <- callback_model_checkpoint(
    filepath = "model.hdf5",
    save_best_only = TRUE,
    period = 1,
    verbose = 1
)

early_stopping <- callback_early_stopping(patience = 5)


history <- model %>%
    fit(X, Y,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_split = 0.2,
        callbacks = list(checkpoint, early_stopping))

## ----inspect model--------------------------------------------------------
plot(history)

save_model_hdf5(model, filepath="unet_model_2_16_256.hdf5")

##model <- load_model_hdf5("unet_model.hdf5", custom_objects=c(dice_coef_loss=dice_coef_loss, dice_coef=dice_coef))

## ----evaluate model-------------------------------------------------------

Y_hat <- predict(model, x = X)

display(combine(Y[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(Y[100,,,], Y_hat[100,,,]), all = TRUE)
display(combine(Y[320,,,], Y_hat[320,,,]), all = TRUE)


##convert to binary and label
Z <- map(array_branch(Y, 1), bwlabel)
Z_hat <- map(array_branch(Y_hat, 1), .f = function(z) bwlabel(z > .5))

display(colorLabels(combine(Z[[10]], Z_hat[[10]])), all = TRUE)
display(colorLabels(combine(Z[[500]], Z_hat[[500]])), all = TRUE)
display(colorLabels(combine(Z[[632]], Z_hat[[632]])), all = TRUE)

##Estimate mean precision
mp <- map2_dbl(Z, Z_hat, mean_precision)
round(mean(mp), 2)
boxplot(mp)

## ----predict test data----------------------------------------------------------

test_data <- tibble(ImageId = dir(TEST_PATH)) %>%
    mutate(ImageFile = file.path(TEST_PATH, ImageId, "images", paste0(ImageId, ".png")),
           ImageShape =  map(ImageFile, .f = function(file) dim(readImage(file))[1:2]),
           X = map(ImageFile, preprocess_image, shape = SHAPE))

test_data %>%
    glimpse()

X <- list2tensor(test_data$X, 4)
Y_hat <- predict(model, x = X)

##compare predicted masks with original imagess

display(combine(X[1,,,], Y_hat[1,,,]), all = TRUE)
display(combine(X[5,,,], Y_hat[5,,,]), all = TRUE)
display(combine(X[32,,,], Y_hat[32,,,]), all = TRUE)

## ----submission----------------------------------------------------------

## construct labelled masks and preform run length encoding and decoding for checking
submission <- test_data %>%
    add_column(Masks = map(array_branch(Y_hat, 1), .f = function(z) bwlabel(z > .5)[,,1])) %>%
    mutate(EncodedPixels = map2(Masks, ImageShape, postprocess_image))

rsamples <- sample_n(submission, 3) %>%
    mutate(Y = map2(EncodedPixels, ImageShape, preprocess_masks, new_shape = SHAPE),
           X = map(ImageFile, preprocess_image, shape = SHAPE)) %>%
    select(X,Y)

X <- list2tensor(rsamples$X, 4)
Y <- list2tensor(rsamples$Y, 4)

display(combine(Y[1,,,], X[1,,,]), all = TRUE)
display(combine(Y[2,,,], X[2,,,]), all = TRUE)
display(combine(Y[3,,,], X[3,,,]), all = TRUE)

submission <- submission %>%
    unnest(EncodedPixels) %>%
    mutate(EncodedPixels = as.character(EncodedPixels)) %>%
    select(ImageId, EncodedPixels)

submission

write_csv(submission, "submission.csv")

