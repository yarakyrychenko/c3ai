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

export_graph <- function(boot, name){
  tg <- data.frame(boot[["typicalGraph"]][["graph"]])
  tg$rownames <- rownames(boot[["typicalGraph"]][["graph"]])
  tg$nodesize <- colSums(boot$typicalGraph$graph)^2 * 16
  write.csv(tg,name)
}

export_graph(boot_ega,"graph_bootega.csv")
print("bootEGA graph saved as graph_bootega.csv")


generate_netloadings <- function(uva, boot_ega, dat, stability_cutoff){
  
  # Step 1: Compute dimension stability
  stability_reduced <- dimensionStability(boot_ega)
  
  # Step 2: Identify stable items
  stable_items <- names(stability_reduced$item.stability$item.stability$empirical.dimensions[
    stability_reduced$item.stability$item.stability$empirical.dimensions > stability_cutoff
  ])
  
  # Step 4: Identify super stable items
  suber_stable_items <- intersect(stable_items, colnames(uva$reduced_data))
  
  # Step 5: Perform EGA on the super stable items
  ega_super_stable <- EGA(data = dat[, suber_stable_items])
  
  # Step 6: Compute network loadings
  netloads.stable <- net.loads(
    data = dat[, suber_stable_items],
    wc = ega_super_stable$wc,
    A = ega_super_stable$network,
    loading.method = "experimental"
  )
  
  return(netloads.stable)
}

get_non_na_rows <- function(mat) {
  non_na_rows <- apply(mat, 1, function(row) any(!is.na(row)))
  mat[non_na_rows, , drop = FALSE]
}

net_threshold <- .9
threshold <- .25

mat_all <- generate_netloadings(uva_all,boot_ega,dat,net_threshold)$std
filtered_mat_all <- ifelse(mat_all  > threshold, mat_all, NA)
nonna_all <- get_non_na_rows(filtered_mat_all)

print("Selected items:")
nrow(nonna_all)
rownames(nonna_all)

dims <- boot_ega_uva[["EGA"]][["dim.variables"]]
write.csv(dims, "dimensions.csv")
print("Dimensions saved as dimensions.csv")