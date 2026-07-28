# model/

Les artefacts du modèle atterrissent ici. Le contenu est **git-ignoré** : ce sont des
sorties d'un export EMaGerLib, pas du code source.

Remplis-le avec :

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export-model.ps1
```

Tu devrais alors y trouver :

| Fichier | Rôle |
|---|---|
| `model_int8.tflite` | entrée du compilateur Neural-ART (doc 4) |
| `emager_model_params.h` | dimensions + scale/zero-point INT8 |
| `emager_prototypes.h` | prototypes d'usine |
| `emager_proto.{h,c}` | classification + calibration sur la carte |
| `emager_selftest.{h,c}` | rejoue les vecteurs de test |
| `emager_test_vectors.h` | les vecteurs de référence |
| `model_fp32.pt` | les poids FP32 exacts de cet export |
| `EXPORT_REPORT.md` | tailles, accord par étape, provenance |
| `generated/` | le code produit par `stedgeai` (doc 4) |

## Les trois fichiers qui doivent rester ensemble

`model_int8.tflite`, `emager_test_vectors.h` et `generated/` décrivent **le même
export**. Régénérer l'un sans les autres produit un firmware qui se vérifie contre un
modèle qu'il n'exécute pas — et le symptôme n'est pas une erreur, ce sont des
embeddings faux. `scripts\export-model.ps1` régénère le `.tflite` et les vecteurs
ensemble pour cette raison ; la recompilation NPU (doc 4) reste à ta charge.

## `model_fp32.pt`

L'entraînement n'est **pas** seedé. Sans ce fichier, les poids qui ont produit un
binaire flashé n'existent plus nulle part et le modèle embarqué devient irreproductible.
Il est ignoré par git parce qu'il est volumineux — mais si tu tagues une version du
firmware, attache-le à la release.
