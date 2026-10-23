# Deep SVDD para detección de anomalías en estructuras cerebrales (THA, PAL, PUT, HIP, AMY)

Un único código y un único conjunto de configs para entrenar el modelo Deep SVDD
(autoencoder de pre-entrenamiento + SVDD) sobre cualquiera de estas estructuras:

| Código | Estructura |
|--------|------------|
| THA | Tálamo |
| PAL | Pálido |
| PUT | Putamen |
| HIP | Hipocampo |
| AMY | Amígdala |

La estructura se elige con `--structure`. Los placeholders `{structure}` de los
configs y los nombres de experimento (`HIP_AE_...`, `THA_AE_...`) se resuelven a
partir de ese parámetro, de modo que los resultados de cada estructura no se pisan.

## Estructura del repositorio

```
src/trainer/           código (DataModule, Models, Training, train_SVDD.py, ...)
configs/               configs YAML compartidos por todas las estructuras
split/{STRUCT}/        splits train/validation/test por estructura
data/labels/           pickle con etiquetas (0 = control, 1 = anómalo)
data/{STRUCT}/hippocampi_images/   máscaras NIfTI (NO están en git, ~14 GB por estructura)
data_preparation/      scripts usados para segmentar OASIS con FastSurfer y armar splits/etiquetas
run_all_structures.sh  entrena varias estructuras una tras otra
```

## Rutas

Los configs y splits usan la raíz `/home/ddeangeli/deepsvdd` (la ruta del servidor
donde se entrenaba). Para correr en otra máquina, `run_all_structures.sh` crea
symlinks `data`, `split` y `configs` dentro de `REMOTE_ROOT` (por defecto
`/home/ddeangeli/deepsvdd`) apuntando a este repositorio. Si preferís otra raíz,
cambiá `REMOTE_ROOT` y reemplazá ese prefijo en `configs/` y `split/`.

## Entrenamiento

Entorno probado: conda `PRIMO` (Python 3.7, PyTorch 1.13, PyTorch Lightning 1.6.4).
Con `pytorch_lightning >= 1.9` falla el import de `VisdomLogger`.

Una estructura:

```bash
cd src/trainer
python train_SVDD.py \
  --configAE ../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml \
  --configSVDD ../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml \
  --structure HIP
```

Varias estructuras en secuencia (un log por estructura en `logs/`, salta las que ya
terminaron, evita que la máquina se suspenda y levanta Visdom en local si hace falta):

```bash
nohup ./run_all_structures.sh > run_all.out 2>&1 &     # THA PAL PUT HIP AMY
./run_all_structures.sh HIP THA                         # solo esas
DRY_RUN=1 ./run_all_structures.sh                       # solo muestra el plan
```

Variables opcionales: `CONDA_ENV`, `CONDA_SH`, `REMOTE_ROOT`, `LOGS_DIR`, `EXTRA_ARGS`.

Los resultados (checkpoints) quedan en `src/results/{STRUCT}_<nombre del config>/`.

## Variante con el volumen como entrada (`VolumeDifference`)

`configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolume.yml` es igual
al config SVDD normal pero con `VolumeDifference: True`: el volumen de cada estructura
(voxels de la máscara) se concatena a las features de su hemisferio antes de la capa
`fcSVDD1`. Los volúmenes se normalizan como z-score con la media y el desvío del conjunto
de entrenamiento (se calculan al inicializar `c` y quedan en el checkpoint).

Para entrenarla reutilizando el autoencoder que ya existe en `src/results` (no se vuelve a
entrenar, así la comparación con la corrida sin volumen es limpia), y sin pisar nada:

```bash
SVDD_CONFIG=AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolume.yml \
SKIP_AE=1 RUN_TAG=_volume nohup ./run_all_structures.sh HIP > run_volume.out 2>&1 &
```

Log en `logs/HIP_volume.log`, resultados en `src/results/HIP_AE_..._16filtersSVDDVolume/`.
Para evaluarla por fuente (OASIS / ADNI / HEC) y contra las referencias sin red:

```bash
cd src/trainer
python ../../tools/evaluate_svdd.py --structure HIP \
  --config AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolume.yml
```

`train_SVDD.py --skip-ae-training` / `SKIP_AE=1` hacen que el SVDD parta del último
checkpoint del autoencoder de esa estructura en lugar de entrenar uno nuevo.

## Variante con el volumen en la salida (`VolumePassthrough`)

`configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolumePass.yml` agrega
`VolumePassthrough: 1.0` (alpha). La salida de la red pasa a ser `[f(x), alpha * v]`, con
`v = (relu(-z_volumen_total), relu(z_asimetria))` calculado a partir de los volúmenes **sin
parámetros** (media/desvío del train, guardados en el checkpoint). El centro `c` tiene 34
dimensiones y el score es `||f - c||^2 + alpha^2 ||v - c_v||^2`.

Por qué fijo y no entrenable: Deep SVDD minimiza `||salida - c||^2`, así que un camino
entrenable del volumen a la salida se aprendería a ignorar. Con `v` fijo, lo que se entrena
(`f`) no cambia y el volumen entra al score con el peso que fija `alpha`.

Para entrenarla con el autoencoder ya existente (rama `local`):

```bash
SVDD_CONFIG=AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolumePass.yml \
SKIP_AE=1 RUN_TAG=_volpass nohup ./run_all_structures.sh HIP > run_volpass.out 2>&1 &
```

`tools/evaluate_svdd.py --config <ese config>` imprime el AUC por fuente de la parte SVDD sola,
de `svdd + a·volumen` para varios pesos, de `volume features only` y de `vol+asym`.
El `alpha` con el que se entrena cambia la escala de `c` pero no el score de la parte `f`.



## Visdom

Los loggers usan Visdom (puerto 8097). El servidor se define en `configs/visdom.yml`
(`server: localhost` por defecto).

## Datos

Las máscaras NIfTI no se versionan. Las de OASIS se generan con
`data_preparation/extract_oasis_structures.py` a partir de la segmentación FastSurfer
(`aseg.auto_noCCseg.mgz`); los splits y las etiquetas con `merge_oasis_splits.py` y
`update_oasis_labels_pickle.py`. Esos scripts tienen rutas absolutas de la máquina
local que hay que adaptar.
