# FoulObs

Compter et observer une foule à partir d'une photo ou d'une courte vidéo — **directement sur votre téléphone ou ordinateur**, sans compte, sans abonnement et **sans envoyer d'image** sur Internet.

## Ce que fait l'appli
- Nombre de personnes visibles (petits groupes : détection personne par personne ; foules denses : carte de densité, marge d'environ 10 %).
- Répartition par zones (2×2, 3×2…), postures (petits groupes), groupes proches.
- Densité en personnes/m² après calibrage de la surface.
- Vidéo : évolution du comptage image par image.
- Rapport à télécharger (Markdown).

FoulObs n'identifie personne : aucun visage, émotion, état de santé ou identité n'est déduit de l'image. Les comptages sont des estimations à vérifier.

## Mise en ligne (GitHub Pages)
Aucun serveur ni clé n'est nécessaire : publiez ce dossier tel quel avec GitHub Pages (Settings → Pages → branche `main`, dossier `/`).
Le premier chargement télécharge les modèles (environ 30 Mo), ensuite ils restent en cache.

## Modèles et licences
- Détection de personnes : COCO-SSD (TensorFlow.js) — Apache 2.0.
- Comptage de foules denses : CSRNet (Li, Zhang & Chen, CVPR 2018), poids `rootstrap-org/crowd-counting` — MIT ; converti en ONNX 8 bits (`csrnet-q8.onnx`).
- Exécution : TensorFlow.js (Apache 2.0), ONNX Runtime Web (MIT), coi-serviceworker (MIT).
