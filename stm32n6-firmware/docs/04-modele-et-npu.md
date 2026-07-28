# 4. Le modèle et le NPU

Du modèle PyTorch entraîné jusqu'à une inférence qui tourne sur la carte — et surtout,
jusqu'à la **preuve** qu'elle calcule la bonne chose.

---

## 4.1 Produire les artefacts (côté PC, dans EMaGerLib)

```powershell
cd C:\dev\EMaGerLib

# le modèle recommandé pour l'embarqué : prototypes calculables sur la carte
python examples\deployment\export_model.py --model EmagerCNNProtoRingStrided

# les données de référence pour vérifier le portage
python examples\deployment\make_test_vectors.py --model EmagerCNNProtoRingStrided
```

Résultat dans `examples\deployment\exported\EmagerCNNProtoRingStrided\` :

| Fichier | À quoi il sert ici |
|---|---|
| `model_int8.tflite` | **l'entrée du compilateur Neural-ART** |
| `emager_model_params.h` | dimensions + scale/zero-point INT8 |
| `emager_prototypes.h` | prototypes d'usine |
| `emager_proto.{h,c}` | classification + calibration sur la carte |
| `emager_selftest.{h,c}` | rejoue les vecteurs de test |
| `emager_test_vectors.h` | les vecteurs eux-mêmes |
| `model_fp32.pt` | les poids exacts — **garde-les** |

> Conserve `model_fp32.pt` avec le firmware. L'entraînement n'est pas seedé : sans ce
> fichier, les poids qui ont produit un binaire flashé n'existent plus nulle part et le
> modèle embarqué devient irreproductible.

Le script `scripts\export-model.ps1` enchaîne les deux commandes et copie tout dans
`model\`.

---

## 4.2 Compiler le réseau pour le NPU

Deux voies, même compilateur derrière.

### Via CubeMX (recommandé pour commencer)

1. Onglet **Software Packs → Select Components** → coche **X-CUBE-AI / Artificial
   Intelligence**.
2. Onglet **Software Packs → X-CUBE-AI** dans l'arbre de gauche → **Add network**.
3. Renseigne :
   - *Type* : **TFLite**
   - *Model* : `...\exported\EmagerCNNProtoRingStrided\model_int8.tflite`
   - *Compression* : **none** — le modèle est déjà en INT8, une compression
     supplémentaire ne ferait que dégrader.
4. **Analyze**. Lis le rapport (voir plus bas).
5. **GENERATE CODE**.

### Via la ligne de commande

Utile pour scripter, ou pour itérer sans ouvrir CubeMX :

```powershell
stedgeai generate --target stm32n6 `
    --model model\model_int8.tflite `
    --name emager --output model\generated
```

`stedgeai analyze --target stm32n6 --model ...` donne le même rapport sans générer.
Le fichier `<name>_generate_report.txt` produit à côté est le document à lire.

---

## 4.3 Lire le rapport — l'étape que tout le monde saute

Le rapport indique quelles parties du graphe tournent **sur le NPU** et lesquelles
retombent **sur le Cortex-M55**. Pour ce modèle, attends-toi à du repli logiciel, et
c'est normal :

- le *ring padding* (padding circulaire sur l'axe des colonnes d'électrodes) se traduit
  en `CONCATENATION` / `SLICE` / `SPLIT` / `TRANSPOSE` ;
- la BatchNorm d'entrée reste **en flottant** (`MUL` / `ADD`) : elle est du côté FP32 du
  QuantStub dans le modèle PyTorch, parce qu'elle normalise la dynamique du signal brut.

Ce n'est pas un problème de correction — le résultat est le même — mais ça change le
raisonnement sur les performances. Si tu comptes publier des chiffres de latence, dis
quelle fraction du graphe est réellement accélérée plutôt que « ça tourne sur le NPU ».

Et rappelle-toi l'ordre de grandeur : ~152 K MAC. Le réseau n'est pas le goulot
d'étranglement. L'acquisition, le fenêtrage et le calcul du MAV dominent.

---

## 4.4 Brancher le réseau à l'application

Le code applicatif de `app/` ne parle jamais directement au runtime d'inférence. Il
passe par un contrat minuscule, [`emager_nn_port.h`](../app/emager_nn_port.h) :

```c
int  emager_nn_init(void);
int  emager_nn_forward(const int8_t in[EMAGER_N_FEATURES],
                       int8_t out[EMAGER_EMBED_DIM]);
```

Deux fonctions, et tout le reste du firmware en dépend sans savoir ce qu'il y a
derrière. C'est ce qui permet de tester la logique applicative avec un faux réseau, et
de changer de runtime sans toucher à l'application.

L'implémentation N6 est [`emager_nn_port_llaton.c`](../app/emager_nn_port_llaton.c).
Elle est **volontairement incomplète** : les appels exacts du runtime LL_ATON dépendent
de ta version de X-CUBE-AI et du nom que tu as donné au réseau, et recopier ici une
signature inventée te ferait perdre plus de temps qu'un `TODO` explicite. Le fichier
indique précisément quoi chercher et où :

- l'en-tête généré `<name>.h` / `ll_aton_rt_user_api.h` dans `model\generated\` ;
- les fonctions `LL_ATON_RT_Init_Network` / `LL_ATON_RT_Main`, et
  `LL_ATON_Set_User_Input_Buffer_Default` / `..._Output_Buffer_Default` pour fournir tes
  propres buffers ;
- l'exemple ST le plus proche : les *getting started* STM32N6 sur
  https://github.com/STMicroelectronics (ImageClassification, ObjectDetection) —
  regarde leur boucle d'inférence, pas leur prétraitement d'image.

Le contrat impose deux choses : entrée et sortie en **INT8**, et retour `0` en cas de
succès. La quantification et la déquantification sont faites par `emager_proto.c` avec
les vraies constantes du modèle — ne les refais pas dans le port.

---

## 4.5 Vérifier que la carte calcule juste

C'est le point le plus important de ce document. Un réseau qui « tourne » et un réseau
qui **calcule la bonne chose** sont deux états très différents, et rien de ce qui
précède ne distingue les deux.

```c
#include "emager_selftest.h"

emager_selftest_result_t r;
emager_prototypes_t factory;
memcpy(&factory, &EMAGER_FACTORY_PROTOTYPES, sizeof(factory));

if (emager_selftest_run(emager_nn_forward, &factory, &r)) {
    printf("selftest OK (%u vecteurs)\n", r.n_vectors);
} else {
    printf("selftest ECHEC: embed %u/%u, classe %u/%u, pire ecart %ld LSB (vecteur %u)\n",
           r.n_embed_ok, r.n_vectors, r.n_class_ok, r.n_vectors,
           (long)r.embed_max_abs_err, r.worst_vector);
}
```

**Lis les deux compteurs séparément** — ils échouent pour des raisons sans rapport, et
les confondre envoie chercher un bug de réseau dans les prototypes :

| Symptôme | Où est le problème |
|---|---|
| embeddings OK, classe fausse | les **prototypes** : blob périmé, drapeaux `valid`, ou carte déjà calibrée |
| embeddings faux | le **réseau** : MAV qui ne correspond pas à l'entraînement (taille de fenêtre, incrément, ordre des canaux), mauvaise quantification d'entrée, ou runtime mal câblé |

La colonne « classe » des vecteurs de test décrit les **prototypes d'usine**. Après une
calibration sur la carte, un écart de classe est attendu et ne signifie rien : c'est
`embed_max_abs_err` qui reste interprétable.

La tolérance est de 2 LSB INT8, pas 0 : le PC utilise les noyaux de référence TFLite, la
carte utilise le NPU. Ils sont d'accord sur l'arithmétique, pas forcément sur le dernier
bit d'arrondi. Exiger l'exactitude bit à bit ferait échouer un portage correct.

Lance ce test **avant** toute calibration, au démarrage, la première fois. Ensuite :
[boot, signature et flash](05-flash-et-boot.md).
