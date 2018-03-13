library(EBImage)

library(reticulate)
py_module_available("skimage.measure")
skim <- import("skimage.measure")

pylabel <- function(image, connectivity = 1) t(skim$label(t(1*imageData(image)), connectivity = connectivity))

y <- matrix(c(0,1,1,0,0,
              0,1,0,0,1,
              0,0,0,1,1,
              0,1,0,1,0,
              1,1,0,1,0), nrow=5, ncol=5, byrow=TRUE)

z <- bwlabel(y)
z
image(z)

zpy <- pylabel(y)
zpy
image(zpy)

all(z == zpy)

x = readImage(system.file("images", "nuclei.tif", package="EBImage"))
x <- x[,,1]
display(x, method = "raster", all = TRUE)

y <- x > .5

display(y, method = "raster", all = TRUE)
z <- bwlabel(y)

display(colorLabels(z), method = "raster")
max(z)

zpy <- pylabel(y, 2)
display(colorLabels(zpy), method = "raster")
max(zpy)
