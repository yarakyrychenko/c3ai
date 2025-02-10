print("Your R libraries are installed in the following paths:")
.libPaths()

library(EGAnet)
library(ggplot2)
dat <- read.csv("data.csv")

uva_all<- UVA(
  data = dat,
  cut.off = 0.25,
  seed = 123
)

print("UVA Results")
print(uva_all)

if (!is.null(uva_all$reduced_data)) {
  print("Using UVA reduced data for EGA")
  ega <- EGA(uva_all$reduced_data)
} else {
  print("Using full data for EGA")
  ega <- EGA(dat)
}

print("EGA Results")
print(ega)

plot_ega <- plot(ega, edge.size = 6)
ggsave("plot_ega.pdf", plot_ega, width=12,height=10) 

print("EGA plot saved as plot_ega.pdf")
