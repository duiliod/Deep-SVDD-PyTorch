# Deep SVDD para detección de anomalías en estructuras cerebrales (THA, PAL, PUT, HIP, AMY)

Un único código y un único conjunto de configs para entrenar el modelo Deep SVDD
(autoencoder de pre-entrenamiento + SVDD) sobre cualquiera de estas estructuras,
usando la cohorte OASIS segmentada con FastSurfer además de los datasets originales:

| Código | Estructura |
|--------|------------|
| THA | Tálamo |
| PAL | Pálido |
| PUT | Putamen |
| HIP | Hipocampo |
| AMY | Amígdala |

Esta rama (`server`) está preparada para el cluster con Slurm. La rama `local` es la
versión para entrenar en una sola máquina.

## Estructura

```
src/trainer/           código (DataModule, Models, Training, train_SVDD.py, ...)
configs/               configs YAML compartidos por todas las estructuras
split/{STRUCT}/        splits train/validation/test por estructura
data/labels/           pickle con etiquetas (0 = control, 1 = anómalo)
data/{STRUCT}/hippocampi_images/   máscaras NIfTI (NO están en git, ~14 GB por estructura)
scripts/train_one.sh   entrena UNA estructura (lo usan Slurm y run_sequential.sh)
scripts/run_sequential.sh   entrena varias estructuras una tras otra (sin Slurm)
scripts/sync_to_server.sh   copia código y/o datos al cluster con rsync
slurm/                 train_structure.sbatch y submit_all.sh
tools/check_data.py    verifica que no falte ningún archivo ni etiqueta
data_preparation/      scripts de la segmentación OASIS (no hacen falta para entrenar)
```

Las rutas de configs y splits usan `{root}` (raíz del repositorio, o `$DEEPSVDD_ROOT`) y
`{structure}` (el valor de `--structure`), así que no dependen de la máquina.

## Puesta en marcha en el cluster

Nodos: `alex` (2x RTX 3060, 8 CPU, 64 GB), `thanos` (2 GPU, 24 CPU, 126 GB) y `loki` (cuando
está levantado). Entorno conda: `~/.conda/envs/deepsvd` (Python 3.7, PyTorch 1.12.1+cu113,
PyTorch Lightning 1.6.4).

1. **Código** (en el cluster; el login node llega a GitHub por HTTPS):
   ```bash
   git clone -b server https://github.com/duiliod/Deep-SVDD-PyTorch.git deepsvdd_oasis
   cd deepsvdd_oasis
   ```
2. **Datos** (desde la máquina que tiene las máscaras; ~14 GB por estructura, ~70 GB en total.
   `/home` del cluster está al 93 %: revisar `df -h /home` antes):
   ```bash
   scripts/sync_to_server.sh data            # las 5 estructuras
   scripts/sync_to_server.sh data HIP THA    # solo algunas
   ```
   Se puede repetir si se corta: rsync solo copia lo que falta.
3. **Verificar** (en el cluster, no necesita GPU):
   ```bash
   ~/.conda/envs/deepsvd/bin/python tools/check_data.py
   ```
   Debe terminar con `ALL OK` y mostrar los tamaños de train/val/test de cada estructura.
4. **Lanzar** un job de Slurm por estructura (1 GPU cada uno):
   ```bash
   slurm/submit_all.sh                # THA PAL PUT HIP AMY
   slurm/submit_all.sh HIP THA        # solo esas
   DRY_RUN=1 slurm/submit_all.sh      # muestra qué enviaría
   ```
   Hay 2 GPUs por nodo, así que con dos nodos arriba entrenan 4 a la vez y la quinta espera.

## Seguimiento

```bash
squeue -u $USER
tail -f slurm_logs/dsvdd_HIP_<jobid>.out
tensorboard --logdir runs/HIP/results      # curvas de pérdida / AUC
```

Cada estructura trabaja en su propia carpeta, así que pueden correr a la vez sin pisarse:

```
runs/<S>/trainer/    directorio de trabajo
runs/<S>/results/    checkpoints y logs de TensorBoard (<S>_AE_..., <S>_AE_...SVDD)
runs/<S>/.done       se crea cuando terminó bien
```

Si un job se corta y se vuelve a enviar, lo que quedó a medias se mueve a `runs/_previous/` y
la estructura se entrena de nuevo desde cero (`lightning.py` carga el autoencoder desde
`version_0`). Las que ya tienen `.done` se saltean (`FORCE=1` para reentrenar).

## Otras opciones

- Varias estructuras en una máquina sin Slurm: `scripts/run_sequential.sh [STRUCT ...]`.
- Prueba rápida: `EXTRA_ARGS="--fast-dev-run --no-pretrained" scripts/train_one.sh HIP`
  (usa `runs/_smoke/`, nunca deja resultados en `runs/<S>`).
- Variables de `scripts/train_one.sh`: `PYTHON_BIN`, `CONDA_ENV`, `CONDA_SH`, `DEEPSVDD_ROOT`,
  `RUNS_DIR`, `CONFIG_DIR`, `EXTRA_ARGS`, `FORCE`.

## Variante con el volumen como entrada (`VolumeDifference`)

`configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDDVolume.yml` es igual
al config SVDD normal pero con `VolumeDifference: True`: el volumen de cada estructura
(voxels de la máscara) se concatena a las features de su hemisferio antes de la capa
`fcSVDD1`. Los volúmenes se normalizan como z-score con la media y el desvío del conjunto
de entrenamiento (se calculan al inicializar `c` y quedan en el checkpoint).

Los scripts de esta rama (`scripts/train_one.sh`, Slurm) todavía usan siempre el config SVDD
normal; el lanzador que permite elegir `SVDD_CONFIG` y `SKIP_AE=1` está en la rama `local`
(`run_all_structures.sh`). Con `train_SVDD.py` se puede lanzar a mano desde `src/trainer`
con `--configSVDD` apuntando al config `...SVDDVolume.yml` y `--skip-ae-training`.
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



## Logs: TensorBoard en lugar de Visdom

Los configs usan `TensorBoardLogger` (`configs/tensorboard.yml`), que solo escribe a disco.
Visdom necesita un servidor accesible desde el nodo de cómputo y desde los nodos del cluster
no hay ruta al que se usaba antes (192.168.7.237). `configs/visdom.yml` sigue en el repo por si
se quiere volver a usarlo (`server:` es configurable); habría que cambiar el `!include` en los
configs `AE_*`.

## Datos

Las máscaras NIfTI no se versionan. Las de OASIS se generan con
`data_preparation/extract_oasis_structures.py` a partir de la segmentación FastSurfer
(`aseg.auto_noCCseg.mgz`); los splits y las etiquetas con `merge_oasis_splits.py` y
`update_oasis_labels_pickle.py`. Esos scripts tienen rutas absolutas de la máquina local.
