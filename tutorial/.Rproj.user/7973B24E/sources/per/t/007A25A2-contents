# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'

hello <- function() {
  print("Hello, world!")
}
### Example of Time Series Analysis Workflow

# Data import, 2 alternatives

myts <- scan()

# Conversion to a time series
mycounts <- ts(ITstore_bidaily$X2, start = 1,
               frequency = 12)

# Alternative with myts vector
mycounts_check <- ts(myts,start = 1,
                     frequency = 12)

# Visualization
plot(mycounts, ylab = "Customer Counts",
     xlab = "Weeks")

library(forecast)
monthplot(mycounts, labels = 1:12,
          xlab = "Bidaily Units")
seasonplot(mycounts, season.labels = F,
           xlab = "")

# Model forecast
plot(forecast(auto.arima(mycounts)))
install.packages('tseries')
plot(lynx); length(lynx)
#keep eye on autocorrelation
plot(LakeHuron); length(LakeHuron)
plot(nottem)
#good seasonality
plot(AirPassengers); length(AirPassengers)
#tren, seasonality and trend within the seasons
plot(EuStockMarkets); length(EuStockMarkets)
#class mts
plot(sunspot.year); length(sunspot.year)
#autocorelation in it, needs more sophisticated method
