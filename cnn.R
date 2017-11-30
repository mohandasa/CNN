# Clean workspace
rm(list=ls())

# Load MXNet
require(mxnet)

# Loading data and set up
#-------------------------------------------------------------------------------

# Load train and test datasets
train_data <- read.csv("C:/Users/anita/train_28.csv")
test_data <- read.csv("C:/Users/anita/test_28.csv")

# Set up train and test datasets
train_data <- data.matrix(train_data)
train_x_data <- t(train_data[, -1])
train_y_data <- train_data[, 1]
train_array_data <- train_x_data
dim(train_array_data) <- c(28, 28, 1, ncol(train_x_data))

test_x_data <- t(test[, -1])
test_y <- test[, 1]
test_array_data <- test_x_data
dim(test_array_data) <- c(28, 28, 1, ncol(test_x_data))

# Set up the symbolic model
#-------------------------------------------------------------------------------

data <- mx.symbol.Variable('data')
# 1st convolutional layer
conv_1_data <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
tanh_1_data <- mx.symbol.Activation(data = conv_1_data, act_type = "tanh")
pool_1_data <- mx.symbol.Pooling(data = tanh_1_data, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))



# 2nd convolutional layer
conv_2_data <- mx.symbol.Convolution(data = pool_1_data, kernel = c(5, 5), num_filter = 50)
tanh_2_data <- mx.symbol.Activation(data = conv_2_data, act_type = "tanh")
pool_2_data <- mx.symbol.Pooling(data=tanh_2_data, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))



# 1st fully connected layer
flattening <- mx.symbol.Flatten(data = pool_2_data)
fc_1_data <- mx.symbol.FullyConnected(data = flattening, num_hidden = 500)
tanh_3_data <- mx.symbol.Activation(data = fc_1_data, act_type = "tanh")


# 2nd fully connected layer
fc_2_data <- mx.symbol.FullyConnected(data = tanh_3_data, num_hidden = 40)
# Output. Softmax output since we'd like to get some probabilities.
try_model <- mx.symbol.SoftmaxOutput(data = fc_2_data)

# Pre-training set up
#-------------------------------------------------------------------------------

# Set seed for reproducibility
mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()

# Training
#-------------------------------------------------------------------------------

# Train the model
model <- mx.model.FeedForward.create(try_model,
                                     X = train_array_data,
                                     y = train_y_data,
                                     ctx = devices,
                                     num.round = 480,
                                     array.batch.size = 40,
                                     learning.rate = 0.01,
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     epoch.end.callback = mx.callback.log.train.metric(100))

# Testing
#-------------------------------------------------------------------------------
mx.ctx.internal.default.value = list(device="cpu",device_id=0,device_typeid=1)
mx.ctx.internal.default.value
class(mx.ctx.internal.default.value) = "MXContext"

# Predict labels
predicted_value <- predict(model, test_array_data)
# Assign labels
predicted_labels_value <- max.col(t(predicted_value)) - 1
# Get accuracy
sum(diag(table(test[, 1], predicted_labels_value)))/40
