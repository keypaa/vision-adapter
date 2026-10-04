# Design — un shard de val packé

> **Réalisé.** `scripts/pack_val.py`, `build_val_plan()` dans
> `vision_adapter/train.py`, et `local_shards` dans `EmbStreamDataset`.
> 305 tests verts. Cette section décrit ce qui a été construit, le reste
> conserve l'historique de la réflexion.

## Correction majeure : le disjoint existait déjà

`scripts/rebuild_val_disjoint.py` construit
`train_manifest_val_disjoint.jsonl` avec `disjoint_val_rows()`, qui filtre par
**key d'embedding** contre le manifest de train — avec un `assert` qui refuse
d'écrire si l'overlap reste. Le val était donc déjà disjoint **par key**, de
façon vérifiée à la construction.

Ce qui est cassé n'est pas le disjoint : c'est que `run_train` reconstruisait un
plan avec `excluded_shards=set(plan.keys())`, ce qui ramenait les 1272 rows à 77
dispersées sur 21 shards. Le disjoint est resté correct pendant tout ce temps ;
c'est le **coût de fetch** qui était le vrai problème.

## Le problème

`run_train` construit le plan de val avec :

```python
val_plan = _build_plan(_val_rows, index,
                       sample_size=len(_val_rows), seed=_plan_seed,
                       excluded_shards=set(plan.keys()))
```

L'exclusion est correcte — une ligne de val ne doit jamais être servie par un
shard que le trainer streame, sinon elle fuit dans le train. Mais la forme
qui en résulte est pathologique :

| | valeur |
|---|---|
| rows du manifest val | 1272 |
| shards du train (exclus) | 80 |
| **rows survivantes** | **77** |
| **shards survivants** | **21** |

Streamer 21 shards pour 77 rows coûte ~21 × 30 s ≈ 10 min par probe, contre
~2 min de training entre deux probes. Sur le run de 2500 steps, 5 probes
ajoutaient ~50 min — presque un doublement du run.

## Pourquoi les deux évidences ne marchent pas

**« Prendre 1-2 shards entiers »** — un fetch, mais un shard est une source de
données unique. C'est un val biaisé, et le biais est invisible : la courbe
monte alors que le modèle se dégrade sur le reste du monde.

**« Garder les 77 rows sur 21 shards »** — diversifié, mais 77 rows ne
mesurent rien. Au mieux ±0,1 nat de bruit, et 10 min pour l'apprendre.

## Le fix : un shard de val packé, construit une fois

Prendre ~1000 rows de val **étalées sur les 21 shards non-exclus**, et les
écrire dans **un seul parquet**.

```
src shards (21)  ──┐
                   ├──>  val_pack.parquet  (1 fichier, ~1000 rows)
src shards (21)  ──┘
```

- **1 fetch** au lieu de 21
- **~1000 rows** au lieu de 77 — 13× moins de bruit
- **tous les domaines représentés** — le biais de source disparaît
- **~2,5 min de stream, une fois.** Puis plus jamais.

## Implémentation

### 1. `scripts/pack_val.py`

Réutilise `pack_rows` — déjà RAM-bounded (`batch_size=64`,
`compression=None` car les payloads bf16 sont incompressibles).

```python
for shard in val_shards:              # les 21 non-exclus
    local = _download_shard_hf_transfer(shard, cache_dir=tmp)
    pf = pq.ParquetFile(local)
    n_take = per_shard                 # 1000 // 21 ≈ 47
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["key", "n_vis", "vis_bytes"])
        #     ^ on ne lit que les colonnes voulues, et un RG à la fois
        for j in range(len(keys)):
            if taken_from_this_shard >= n_take:
                break
            ...
```

Points à respecter :

- **Une ligne = une row du parquet source**, décodée comme le fait
  `stream.py` : `np.frombuffer(vis_bytes, uint8).view(bfloat16).reshape(-1, 4096).float()`
  puis ré-encodée par `make_row`. Ne jamais materialiser le shard entier.
- **Le sampling est par shard**, pas global : 47 rows par shard, en prenant des
  row groups **espacés** (pas les 47 premières, qui seraient tous du même
  bucket de `n_vis`).
- **Écrire les `key` d'origine**, pas des noms synthétiques — sinon le key
  index ne résout pas les rows au moment de la lecture.

### 2. Format de sortie

`SCHEMA` de `pack.py` : `key: string`, `n_vis: int64`, `vis_bytes: binary`.
Un seul fichier : `emb_cache/val_pack.parquet`.

Le manifest val reste utilisé pour le **texte** (`user`, `assistant`,
`grid_thw`, `g`) ; le pack ne porte que les embeddings. Les deux se rejoignent
par `key`.

### 3. Branchement dans `train.py`

```python
val_pack = data_dir / "val_pack.parquet"
if val_pack.is_file():
    # chemin rapide : 1 shard, pas de plan, pas d'exclusion à vérifier
    val_order = ["val_pack.parquet"]
    val_plan = {"val_pack.parquet": [rows matching the pack's keys]}
else:
    # le chemin actuel, 21 shards — kept as the fallback
    ...
```

La garde `excluded_shards` disparaît **dans ce cas seulement** : par
construction le pack est fait de shards exclus du train, donc l'exclusion est
déjà satisfaite. Il faut quand même le **vérifier** au démarrage et refuser de
charger un pack dont une key pointe vers un shard du train — sinon un pack
reconstruit à la main après un changement de plan de train pourrait fuiter.

### 4. Tests

- le pack contient bien des keys de tous les shards non-exclus
- aucune key du pack ne pointe vers un shard du plan de train
- un pack dont `n_vis` ne correspond pas au `grid_thw` de sa row est rejeté
  (le mismatch guard de `core.py` le prend déjà, mais il faut le pinner)
- `pack_rows` écrit et relit le même nombre de rows

## Impact attendu

| | avant | après |
|---|---|---|
| fetch par probe | 21 shards | 1 shard |
| rows par probe | 77 | ~1000 |
| coût par probe | ~10 min | ~30 s |
| sur 2500 steps, 5 probes | ~50 min | ~3 min |

Avec ça, `val_every=250` devient affordable — la val est un forward sans
streaming derrière.

## Le compte de rows change — et il doit changer explicitement

Retirer 1000 rows du manifest de train change les dénominateurs :

| | avant | après |
|---|---|---|
| rows au manifest | 117 600 | 116 600 |
| **rows embeddables** | **116 435** | **115 435** |

Sur le seuil Baseten (57 600 samples) c'est négligeable — mais c'est **1 % du
ratio epoch**, et 1 % de dérive sur un dénominateur est exactement ce qui fait
qu'on lit « 1,00 epoch » dans un log qui en vaut 1,01.

**Le compte doit être pris après le retrait, jamais avant.** `build_epoch_plan`
calcule déjà `n_available` après avoir filtré l'index — c'est le bon endroit.
Le risque est ailleurs : si le header du run lit la taille du manifest brut
(117 600) pendant que le plan en contient 116 435 après exclusions, le
`samples_seen / n_rows` affiché est faux.

À faire donc :
1. que le header du run annonce **le compte effectif du plan**, avec la source
   (manifest brut, après embedding, après exclusions val)
2. un test qui vérifie que retirer N keys du manifest décaisse bien le compte de
   N — sinon le ratio reste fantaisiste en silence

## Alternative écartée : `whole_shards=True` sur `build_epoch_plan`

Plus simple (une option), mais elle rend le val **dépendant du plan de train** :
si le train change de shards, la taille et la composition du val changent avec,
et deux runs ne sont plus comparables. Un pack figé est un point de comparaison
stable — c'est la seule raison qui compte quand on veut comparer deux
checkpoints.