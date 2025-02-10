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
  print("Using UVA reduced data for bootEGA")
  boot_ega <- bootEGA(
  data = uva_all$reduced_data,
  seed = 1, 
  iter = 500
)
} else {
  print("Using full data for bootEGA")
  boot_ega <- bootEGA(
  data = dat,
  seed = 1, 
  iter = 500
)
}

print("bootEGA Results")
print(boot_ega)

boot_ega_plot <- plot(boot_ega, edge.size = 6, node.size = colSums(boot_ega$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("plot_bootega.pdf", boot_ega_plot, width=12,height=10) 

print("bootEGA plot saved as plot_bootega.pdf")
