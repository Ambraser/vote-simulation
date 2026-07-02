#Fonction qui crée une copule gaussienne et la transforme en uniforme

copule_gaussienne_unif <- function(nb_candidats, nb_electeurs, R) {
  votes <- MASS::mvrnorm(
    n = nb_electeurs,
    mu = rep(0, nb_candidats),
    Sigma = R
  )
  vunif <- t(apply(votes, 2, pnorm))
  return(vunif)
}


#Premièrement, on "fixe" la dépendance par une matrice variance covariance random
#Quand je dis "fixe" c'est la méthode de génération de dépendance que je fixe, la matrice en elle même est aléatoire
#C'est seulement qu'il existe d'autres méthodes de génération de dépendance (négatives, polarisées, etc...)

##Pour les lois marginales des candidats, elles doivent être différentes pour chaque candidats
#Ici on prend une loi beta dont les paramètres sont tirés différemment pour tous les candidats
#a et b tirés aléatoirement uniformément entre 0 et 3

eval_ddd_beta <- function(nb_candidats, nb_electeurs, K = 1, alpha_min = 0, alpha_max = 3, beta_min = 0, beta_max = 3) {
  Phi_ddd <- vector("list", K)
  if (alpha_min <=0 ){
    alpha_min <- 1e-6
  }
  if (beta_min <=0 ){
    beta_min <- 1e-6
  }
  if (alpha_max <=0 || beta_max <=0 ){
    alpha_max <- 3
    beta_max <- 3
  }
  eps <- 1e-6
  for (j in 1:K) {
    R <- randcorr(nb_candidats)
    vunif <- copule_gaussienne_unif(nb_candidats, nb_electeurs, R)
    phi <- matrix(0, nrow = nb_candidats, ncol = nb_electeurs)
      for (i in 1:nb_candidats) {
        a <- runif(1, alpha_min, alpha_max)
        b <- runif(1, beta_min, beta_max)
      phi[i, ] <- qbeta(
        vunif[i, ],
        shape1 = a,
        shape2 = b
      )
    }
    Phi_ddd[[j]] <- phi
  }
  return(Phi_ddd)
}


#Ici on force a=b et ils sont tirés entre 0 et 1
#Ca donne donc des candidats toujours assez polarisés
#Exemple : très appréciés ET très detesté, ou un peu apprecié et un peu detesté
#Toujours autant apprecié que detesté car a=b

eval_ddd_beta_polar <- function(nb_candidats, nb_electeurs, K = 1) {
  Phi_ddd <- vector("list", K)
  eps <- 1e-6
  for (j in 1:K) {
    R <- randcorr(nb_candidats)
    vunif <- copule_gaussienne_unif(nb_candidats, nb_electeurs, R)
    phi <- matrix(0, nrow = nb_candidats, ncol = nb_electeurs)
    for (i in 1:nb_candidats) {
      a <- runif(1, eps, 0.5)
      b <- a
      phi[i, ] <- qbeta(
        vunif[i, ],
        shape1 = a,
        shape2 = b
      )
    }
    Phi_ddd[[j]] <- phi
  }
  return(Phi_ddd)
}




#Ces deux générations de données par evaluation couvre une large zone du cube indice evaluation