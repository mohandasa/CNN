install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
library(mxnet)


# Clear workspace
rm(list=ls())

# Load EBImage library
library(EBImage)
require(EBImage)

# Load data
X_data <- read.csv("C:/Users/anita/olivetti_X.csv", header = F)
labels <- read.csv("C:/Users/anita/olivetti_y.csv", header = F)

# Dataframe of resized images
rs_data <- data.frame()

# Main loop: for each image, resize and set it to greyscale
for(i in 1:nrow(X_data))
{
  # Try-catch
  result <- tryCatch({
    # Image (as 1d vector)
    imgset <- as.numeric(X_data[i,])
    # Reshape as a 64x64 image (EBImage object)
    imgset <- Image(imgset, dim=c(64, 64), colormode = "Grayscale Image")
    # Resize image to 28x28 pixels
    img_resized_data <- resize(img, w = 28, h = 28)
    # Get image matrix (there should be another function to do this faster and more neatly!)
    img_matrix_data <- img_resized_data@.Data
    # Coerce to a vector
    img_vectors <- as.vector(t(img_matrix_data))
    # Add label
    label <- labels[i,]
    var1 <- c(label, img_vectors)
    # Stack in rs_df using rbind
    rs_data <- rbind(rs_data, var1)
    # Print status
    print(paste("Executed",i,sep = " "))},
    # Error function (just prints the error). Btw you should get no errors!
    error = function(e){print(e)})
}


# Set names. The first columns are the labels, the other columns are the pixels.
names(rs_data) <- c("label", paste("pixel", c(1:784)))

# Train-test split
#-------------------------------------------------------------------------------
# Simple train-test split. No crossvalidation is done in this tutorial.

# Set seed for reproducibility purposes
set.seed(100)

# Shuffled df
shuffled <- rs_data[sample(1:400),]

# Train-test split
train_28_data <- shuffled[1:360, ]
test_28_data <- shuffled[361:400, ]

# Save train-test datasets
write.csv_train(train_28, "C://train_28.csv", row.names = FALSE)
write.csv_test(test_28, "C://test_28.csv", row.names = FALSE)

# Done!
print("Executed!")