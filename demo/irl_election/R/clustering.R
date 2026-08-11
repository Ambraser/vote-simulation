###################################################
# Calculs des clusters méthodes de vote #
# juin 2026                        #
###################################################
rm(list = ls())
library(MASS)
library(FactoMineR)
path_image<-"images clustering/"

path_data<-"res/res/MODEL_UNI/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_UNI_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_0-05/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_05_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_1/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_1_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_0-5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_2/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_2_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances, 9), choice="map")
dev.off()
}


###
out<-c(3,21,22,26,32,33)

path_image<-"images clustering out/"

path_data<-"res/res/MODEL_UNI/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_UNI_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_0-05/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_05_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_1/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_1_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}


path_data<-"res/res/MODEL_DDD_BETA_0-5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_2/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_2_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[-out,-out], 6), choice="map")
dev.off()
}



###################################################
condo<-c(4,5,9,10,11,17,18,24,27)
path_image<-"images clustering condo/"
colnames(distances)

path_data<-"res/res/MODEL_UNI/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_UNI_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}

condo<-c(4,5,9,10,11,17,18,24,27,33)

path_data<-"res/res/MODEL_DDD_BETA_0-05/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_05_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_1/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_1_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_0-5/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_0_5_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}

path_data<-"res/res/MODEL_DDD_BETA_2/csv/"
cand<-c(3,4,5,6,8,10,15)
for (i in cand)
{distances <-read.csv(file=paste0(path_data,"mean_c",i,"_distance.csv"), row.names = 1)
png(file=c(paste0(path_image,"mds_BETA_2_clus_c",i,".png")),width = 1000, height = 1000)
plot(HCPC(distances[condo,condo], 6), choice="map")
dev.off()
}



###################################################
distances <-read.csv2(file="vote_CSES.csv", row.names = 1)
png(file="CSES.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="map")
dev.off()

distances <-read.csv2(file="vote_CSES.csv", row.names = 1)
png(file="CSES_dendo.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="tree")
dev.off()

distances <-read.csv2(file="vote_CSES.csv", row.names = 1)
png(file="CSES_3D.png",width = 600, height = 600)
plot(HCPC(distances, 6), choice="3D.map")
dev.off()

df<-read.csv("cses_data_/cleaned_BELW2019_cses.csv")
apply(df, 2, mean)
apply(df, 2, hist)
