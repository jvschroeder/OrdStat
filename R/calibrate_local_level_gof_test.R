#' Calibrates the the goodness-of-fit test with equal local levels proposed by Gontscharuk et al. (2016)
#' 
#' Calibrates the the goodness-of-fit test with equal local levels proposed by
#'  Gontscharuk, V., Landwehr, S., Finner, H., (2016). Goodness of fit tests in
#'  terms of local levels with special emphasis on higher criticism tests.
#'  Bernoulli 22, 1331-1363. doi:10.3150/14-BEJ694.
#' 
#' @param n The number of hypotheses
#' @param alpha The desired global level
#' 
#' @return Returns the local level necessary to achieve the desired global level
#' 
#' @export
calibrate_local_level_gof_test <- function(n,alpha) {
  interval <- c(alpha/n,alpha)
  i <- 1:n
  
  expectation <- function(v) 1-pordstat2(1-rev(qbeta(v,i,n-i+1)),function(x) x,n,0,progress=F)[n+1]
  
  for(j in 1:1000) {
    e <- expectation(mean(interval))
    if(e > alpha) {
      interval[2] <- mean(interval)
    } else {
      interval[1] <- mean(interval)
    }
    if( diff(interval) < .Machine$double.eps) break
  }
  if(j == 1) warning("The calibration did not fully converge within 1000 steps")
  return(mean(interval))
}