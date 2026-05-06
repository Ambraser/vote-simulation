# Cahier des charges — Interface graphique `vote_simulation`

## 1. Contexte et objectifs

`vote_simulation` est un framework Python de simulation de scrutins électoraux.
Le workflow complet repose aujourd'hui sur un fichier TOML de configuration et des appels Python directs.
L'objectif est de fournir une **interface web interactive** — inspirée de R Shiny — permettant à un utilisateur non-développeur de configurer et lancer le pipeline complet sans toucher au code ni aux fichiers de configuration manuellement.

---

## 2. Rappel du workflow cible

```
┌─────────────────────────────────────────────┐
│            simulation.toml                  │
│                                             │
│  [simulation]                               │
│  output_base_path = "../data/"              │
│  generative_models = ["VMF_HC", "IC"]       │
│  rule_codes       = ["PLU1", "BORD", ...]   │
│  candidates       = [3, 14]                 │
│  voters           = [11, 101, 1001]         │
│  iterations       = 1000                    │
│  seed             = 42                      │
│                                             │
│  [generator_params.VMF_HC]                  │
│  vmf_concentration = 10.0                   │
└──────────────────┬──────────────────────────┘
                   │
         simulation_from_config()
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
  generate_data()       run_rules_on_instance()
       │                       │
  data/gen/              data/sim_result/
  <MODEL>_v<NV>_c<NC>/   <MODEL>_v<NV>_c<NC>/
  iter_0001.parquet      iter_0001.parquet
  ...                    ...
```

Le fichier TOML est la **source de vérité** de toute simulation.
L'interface graphique doit permettre de **le générer, le modifier et le sauvegarder**, puis lancer les fonctions Python correspondantes.

---

## 3. Propositions de librairies

### 3.1 Comparatif des frameworks UI

| Critère | **Gradio** | **Panel** | **Streamlit** | **Dash** |
|---|---|---|---|---|
| Installation | `pip install gradio` | `pip install panel` | `pip install streamlit` | `pip install dash` |
| Lancement | `gr.launch()` | `pn.serve()` | `streamlit run app.py` | Serveur Flask |
| Syntaxe | Déclarative, Shiny-like | Widgets dynamiques | Script linéaire | Composants React |
| Composants | Checkboxes, dropdowns, progress, file upload | Idem + plots natifs | Idem | Idem + graphiques |
| Compat. notebook | ✅ `inline=True` | ✅ natif | ⚠️ limité | ❌ |
| Callbacks temps réel | ✅ générateurs Python | ✅ | ✅ | ✅ |
| Apprentissage | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |

> **Recommandation : Gradio**  
> Démarrage en une ligne, syntaxe proche de R Shiny (reactive inputs), compatible avec les notebooks `demo/*.ipynb` via `gr.launch(inline=True)`, et ne nécessite pas de serveur externe.

### 3.2 Librairies complémentaires recommandées

| Usage | Librairie | Déjà dans `pyproject.toml` |
|---|---|---|
| Visualisation — heatmaps, histogrammes | `matplotlib` | ✅ |
| Dataframes interactifs | `pandas` | ✅ |
| Lecture / écriture TOML | `tomllib` (stdlib Python 3.11+) + `tomli-w` | ❌ à ajouter |
| Lecture fichiers Parquet | `pyarrow` | ✅ |
| Barres de progression | `tqdm` (déjà utilisé dans la simulation) | ✅ |
| Heatmap avancée (optionnel) | `seaborn` | ❌ optionnel |

Ajout minimal dans `pyproject.toml` :
```toml
[project.optional-dependencies]
ui = [
    "gradio>=4.0",
    "tomli-w>=1.0",   # écriture de fichiers TOML
]
```

Installation : `pip install -e ".[ui]"`

---

## 4. Lancement

```bash
# Après installation du package avec extras UI
pip install -e ".[ui]"

# Lancement de l'interface
vote-sim-ui

# Ou directement
python -m vote_simulation.ui
```

Le navigateur par défaut s'ouvre automatiquement sur `http://localhost:7860`.

Entry-point à ajouter dans `pyproject.toml` :
```toml
[project.scripts]
vote-sim-ui = "vote_simulation.ui:launch"
```

---

## 5. Structure de l'interface — 4 onglets

```
┌─────────────────────────────────────────────────────────┐
│  vote_simulation  │ Config │ Données │ Simulation │ Résultats │
│                   └────────┴─────────┴────────────┴───────────┘
│  [Statut : Prêt]          [Fichier TOML actif : config/simulation.toml]
│  [▶ Run complet]
└─────────────────────────────────────────────────────────┘
```

---

### Onglet 1 — Configuration

**But** : Charger ou construire la configuration TOML sans l'éditer manuellement.

#### Paramètres exposés (mappent directement sur `[simulation]`)

| Composant UI | Type | Clé TOML | Valeur par défaut |
|---|---|---|---|
| Dossier de sortie | Champ texte | `output_base_path` | `"../data/"` |
| Seed | Champ entier | `seed` | `42` |

#### Actions

| Bouton | Comportement |
|---|---|
| **Charger un TOML** | Upload d'un fichier `.toml` → remplit tous les champs de tous les onglets |
| **Exporter la config** | Génère et télécharge `simulation.toml` à partir des champs courants (via `tomli-w`) |
| **Réinitialiser** | Recharge les valeurs par défaut |

> Le fichier TOML exporté respecte exactement la structure attendue par `load_simulation_config()`,
> y compris la section `[generator_params.<MODEL>]` si des paramètres avancés sont renseignés.

---

### Onglet 2 — Génération de données

**But** : Configurer les paramètres de génération et lancer `generate_data()`.

#### 2.1 Modèles génératifs (`generative_models`)

**CheckboxGroup** peuplé dynamiquement via `list_generator_codes()` au démarrage :

```
☐ UNI        ☐ IC         ☐ IANC
☐ EUCLID     ☐ EUCLID_1D  ☐ EUCLID_2D   ☐ EUCLID_3D   ☐ EUCLID_5D
☐ GAUSS      ☐ LADDER     ☐ SPHEROID    ☐ PERTURB
☐ UNANIMOUS  ☐ UFR        ☐ VMF_HC      ☐ VMF_HS
```

#### 2.2 Paramètres avancés par modèle (`[generator_params.<MODEL>]`)

Panneau conditionnel affiché lorsque le modèle correspondant est coché :

| Modèle | Paramètre | Type | Méppe sur |
|---|---|---|---|
| `VMF_HC` / `VMF_HS` | `vmf_concentration` | Float (défaut `10.0`) | `[generator_params.VMF_HC] vmf_concentration` |
| `EUCLID` | `box_dimensions` | Liste de floats, ex. `[1.0, 1.0]` | `[generator_params.EUCLID] box_dimensions` |
| `GAUSS` | `sigma` | Liste de floats, ex. `[1.0]` | `[generator_params.GAUSS] sigma` |
| `LADDER` | `n_rungs` | Entier (défaut `21`) | `[generator_params.LADDER] n_rungs` |
| `SPHEROID` | `stretching` | Float (défaut `1.0`) | `[generator_params.SPHEROID] stretching` |

#### 2.3 Combinaisons de simulation

| Composant | Type | Clé TOML | Exemple |
|---|---|---|---|
| Nombre de votants | Tag input (liste d'entiers) | `voters` | `11, 101, 1001` |
| Nombre de candidats | Tag input (liste d'entiers) | `candidates` | `3, 14` |
| Nombre d'itérations | Slider (1 – 10 000) | `iterations` | `1000` |

**Indicateur temps réel** :  
`"Profils à générer : 2 modèles × 3 voters × 2 candidats × 1000 itérations = 12 000 profils"`

#### 2.4 Actions

| Bouton | Comportement |
|---|---|
| **Générer les données** | Appelle `generate_data(toml_temp_path)` |
| **Annuler** | Interrompt la génération (thread séparé) |

Feedback :
- Barre de progression (wrappant le `tqdm` interne)
- Zone de logs scrollable (paths générés, erreurs)
- Résumé final : nb de fichiers, taille totale sur disque

---

### Onglet 3 — Simulation (règles de vote)

**But** : Sélectionner les règles et lancer `simulation_from_config()`.

#### 3.1 Sélection des règles (`rule_codes`)

**CheckboxGroup** peuplé dynamiquement via `get_all_rules_codes()`, organisé par famille :

| Famille | Codes |
|---|---|
| **Pluralité / IRV** | `PLU1`, `PLU2`, `HARE`, `IRV`, `IRVA`, `IRVD`, `SIRV`, `ICRV`, `BUCK_I`, `BUCK_R` |
| **Condorcet** | `COPE`, `SCHU`, `MMAX`, `BLAC`, `CAIR`, `CVIR`, `TIDE`, `KIMR`, `KEME`, `SLAT`, `SPCY`, `WOOD`, `YOUN`, `DODG_C`, `DODG_S` |
| **Score / Borda** | `BORD`, `COOM`, `NANS`, `CSUM`, `RPAR`, `EXHB`, `BALD` |
| **Jugement / Score continu** | `MJ`, `RV`, `STAR`, `VETO` |
| **Approbation — seuil** | `AP_T`, `AP_T05`, `AP_T06`, `AP_T07`, `AP_T08`, `AP_T09` |
| **Approbation — K** | `AP_K`, `AP_K2` … `AP_K12` |
| **Pro. Veto** | `PV-IR`, `PV-BALD`, `PV-COMB`, `PV-NANS`, `PV-CIRV`, `PV-DAUN`, `PV-BMF` |

Boutons rapides :

| Bouton | Comportement |
|---|---|
| **Tout sélectionner** | Coche toutes les règles |
| **Désélectionner** | Décoche tout |
| **Jeu rapide (15 règles)** | Sélectionne `PLU1, HARE, BORD, COOM, BALD, COPE, SCHU, MMAX, PV-IR, PV-BALD, PV-COMB, PV-NANS, PV-CIRV, PV-DAUN, PV-BMF` |
| **Jeu SVVAMP complet (55 règles)** | Sélectionne toutes les règles SVVAMP |

#### 3.2 Source des données

Radio button :
- **Utiliser les données générées** (onglet 2) → lit `output_base_path/gen/`
- **Charger un dossier existant** → champ texte `input_folder_path`

#### 3.3 Actions

| Bouton | Comportement |
|---|---|
| **Lancer la simulation** | Appelle `simulation_from_config(toml_temp_path)` |
| **Annuler** | Interrompt le traitement |

Feedback :
- Barre de progression globale
- Logs : modèle / voters / candidats / itération en cours
- Résumé final : nb de règles × profils traités, chemins des fichiers résultats

---

### Onglet 4 — Résultats

**But** : Explorer les résultats Parquet sans quitter l'interface.

#### 4.1 Sélecteur

Dropdowns chaînés (peuplés en scannant `data/sim_result/`) :

```
Modèle : [IC ▼]   Voters : [1001 ▼]   Candidats : [14 ▼]
```

Correspondance avec la structure de dossiers :
```
data/sim_result/IC_v1001_c14/iter_0001.parquet
                              iter_0002.parquet ...
```

#### 4.2 Visualisations disponibles

| Vue | Description |
|---|---|
| **Tableau de fréquences** | DataFrame : règle × candidat → % de victoires sur toutes les itérations |
| **Heatmap d'accord** | Matrice règle × règle : fréquence à laquelle deux règles désignent le même gagnant |
| **Distribution co-gagnants** | Histogramme : nombre moyen de co-gagnants par règle |

#### 4.3 Export

| Bouton | Comportement |
|---|---|
| **Exporter CSV** | Télécharge le tableau de fréquences courant |
| **Exporter PNG** | Télécharge le graphique actuellement affiché |

---

## 6. Barre globale (toujours visible)

```
┌──────────────────────────────────────────────────────────────────┐
│  ● Prêt   │  TOML actif : config/simulation.toml   │  [▶ Run complet]  │
└──────────────────────────────────────────────────────────────────┘
```

| Élément | Description |
|---|---|
| Indicateur d'état | `Prêt` / `Génération en cours…` / `Simulation en cours…` / `Terminé ✓` |
| Chemin TOML actif | Mis à jour à chaque modification / import |
| **Bouton Run complet** | Enchaîne Génération → Simulation en appelant directement `simulation_from_config()` |

---

## 7. Structure des fichiers à créer

```
src/vote_simulation/
└── ui/
    ├── __init__.py          # expose launch()
    ├── app.py               # gr.Blocks principal, barre globale, onglets
    ├── tab_config.py        # Onglet 1 — Config / TOML
    ├── tab_generation.py    # Onglet 2 — Génération de données
    ├── tab_simulation.py    # Onglet 3 — Règles de vote
    ├── tab_results.py       # Onglet 4 — Visualisation des résultats
    └── toml_utils.py        # Lecture / écriture TOML ↔ état Gradio
```

`toml_utils.py` gère la **bijection entre l'état de l'UI et un fichier TOML valide** :

```python
# toml_utils.py (schéma)

def state_to_toml(state: dict) -> str:
    """Convertit l'état courant de l'UI en contenu TOML valide."""
    # Construit la section [simulation] + [generator_params.*]
    ...

def toml_to_state(toml_path: str) -> dict:
    """Parse un fichier TOML et retourne l'état initial de l'UI."""
    ...
```

---

## 8. Exigences non-fonctionnelles

| Exigence | Détail |
|---|---|
| **Facilité de lancement** | Une seule commande (`vote-sim-ui`), pas de configuration préalable |
| **Pas de duplication de logique** | L'UI appelle directement `simulation_from_config()`, `generate_data()`, `list_generator_codes()`, `get_all_rules_codes()` |
| **Lecture dynamique des registres** | Les listes de règles et de générateurs sont lues au démarrage, pas codées en dur |
| **TOML comme source de vérité** | Toute modification dans l'UI se reflète dans le TOML ; le TOML importé remplit l'UI |
| **Compatibilité notebook** | `gr.launch(inline=True)` pour intégration dans `demo/*.ipynb` |
| **Reproductibilité** | Le fichier TOML exporté est relançable à l'identique en CLI via `simulation_from_config()` |
| **Feedback temps réel** | Progression wrappant le `tqdm` interne de la simulation |

---

## 9. Priorités de développement

| Priorité | Fonctionnalité |
|---|---|
| **P0** | `toml_utils.py` : lecture/écriture TOML ↔ UI |
| **P0** | Onglet Config : import / export TOML |
| **P0** | Onglet Génération + Simulation : sélection + bouton Run complet |
| **P1** | Barre de progression temps réel |
| **P1** | Onglet Résultats : tableau de fréquences |
| **P2** | Paramètres avancés par modèle (VMF, EUCLID, etc.) |
| **P2** | Heatmap d'accord entre règles |
| **P3** | Distribution des co-gagnants, export PNG |
