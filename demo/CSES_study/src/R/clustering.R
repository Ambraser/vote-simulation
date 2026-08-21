###################################################
# Calculs des clusters méthodes de vote #
# juin 2026                        #
###################################################

###################################################
rm(list = ls())
library(MASS)
library(FactoMineR)
path_image<-"images clustering/"
distances <-read.csv(file="vote_CSES.csv", row.names = 1)
distances
png(file="CSES.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="map")
dev.off()

distances <-read.csv(file="vote_CSES.csv", row.names = 1)
png(file="CSES_dendo.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="tree")
dev.off()

distances <-read.csv(file="vote_CSES.csv", row.names = 1)
png(file="CSES_3D.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="3D.map")
dev.off()

df<-read.csv("cses_data_/cleaned_BELW2019_cses.csv")
apply(df, 2, mean)
apply(df, 2, hist)

