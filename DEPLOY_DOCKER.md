# Deployment Docker

Ce guide couvre un deploiement complet et exploitable en production legere:
- redemarrage automatique apres crash
- exposition publique via IP et port
- mise a jour depuis git
- limites CPU/memoire/processus
- separation stricte code et data utilisateur
- execution de vote-sim-ui dans le conteneur

## 1) Prerequis

1. Docker Engine doit etre actif.
2. Le CLI Docker doit pointer vers un daemon actif.

Verifier le contexte actif:

```bash
docker context ls
```

Si Docker Desktop est eteint mais Docker Engine Linux est actif, basculer vers le contexte local:

```bash
docker context use default
```

Verifier la connexion daemon:

```bash
docker info
```

## 2) Build et lancement

### Cas standard (port 8501 libre)

```bash
docker compose -f docker-compose.yml up -d --build vote-sim-ui
```

Application accessible sur:
- http://localhost:8501
- http://<IP_PUBLIQUE_SERVEUR>:8501

### Cas port 8501 deja occupe

Le compose accepte un port hote configurable via la variable VOTE_SIM_HOST_PORT.

Exemple avec 8502:

```bash
VOTE_SIM_HOST_PORT=8502 docker compose -f docker-compose.yml up -d --build vote-sim-ui
```

Application accessible sur:
- http://localhost:8502
- http://<IP_PUBLIQUE_SERVEUR>:8502

## 3) Contraintes techniques implementees

- Restart on crash: restart: unless-stopped
- IP publique exposee: mapping ${VOTE_SIM_HOST_PORT:-8501}:8501 et ecoute Streamlit sur 0.0.0.0
- Update depuis git: script deploy/update_from_git.sh
- Limites ressources: mem_limit: 4g, cpus: 2.0, pids_limit: 256
- Separation code/data: volume ./docker-data:/data
- Commande executee dans le conteneur: vote-sim-ui
- Healthcheck HTTP: /_stcore/health

## 4) Mise a jour depuis git

```bash
./deploy/update_from_git.sh
```

Ce script:
1. synchronise git (fetch + pull --ff-only)
2. rebuild l image Docker
3. redemarre le service

Si tu utilises un port alternatif, relance ensuite avec la variable:

```bash
VOTE_SIM_HOST_PORT=8502 docker compose -f docker-compose.yml up -d vote-sim-ui
```

## 5) Donnees utilisateur

- Le repertoire /data dans le conteneur est persistant via ./docker-data.
- L application est preconfiguree avec VOTE_SIM_OUTPUT_BASE_PATH=/data.
- Les resultats utilisateur survivent aux rebuilds/restarts.

## 6) Verification et exploitation

Etat des services:

```bash
docker compose -f docker-compose.yml ps
```

Logs:

```bash
docker compose -f docker-compose.yml logs -f vote-sim-ui
```

Arret:

```bash
docker compose -f docker-compose.yml down
```

## 7) Exposition internet

1. Ouvrir le port publie (8501 par defaut, sinon celui de VOTE_SIM_HOST_PORT) dans firewall/NSG.
2. Si reverse proxy (Nginx/Traefik), router vers localhost:<PORT_HOTE>.
3. Recommande: mettre HTTPS au niveau reverse proxy.
