\name{OrdStat-package}
\alias{OrdStat-package}
\alias{OrdStat}
\docType{package}
\title{
  Calculate the joint distribution of order statistics
}
\description{
  Calculate the joint distribution of order statistics for the one- or two-group case.
}
\details{
}
\author{
Jonathan von Schroeder

Maintainer: Jonathan von Schroeder <jvs@uni-bremen.de>
}
\references{
}
\keyword{ package }
\seealso{
}
\examples{
  \dontrun{
    t <- 0.05 * (1:50)/50
    noe_faithful(t,pnorm(qnorm(t),mean=1))  
  }
}
