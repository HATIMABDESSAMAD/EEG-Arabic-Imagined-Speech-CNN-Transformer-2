# Deploiement Streamlit Cloud

## 📦 Déploiement rapide

### Option 1: Via l'interface web Streamlit Cloud

1. **Connectez-vous à Streamlit Cloud**
   - Allez sur [share.streamlit.io](https://share.streamlit.io)
   - Connectez-vous avec votre compte GitHub

2. **Créer une nouvelle application**
   - Cliquez sur "New app"
   - Sélectionnez votre dépôt: `HATIMABDESSAMAD/EEG-Arabic-Imagined-Speech-CNN-Transformer`
   - Branch: `main`
   - Main file path: `app.py`

3. **Configuration avancée (optionnel)**
   - Python version: 3.10 (recommandé)
   - Cliquez sur "Advanced settings" si vous voulez personnaliser

4. **Déployer**
   - Cliquez sur "Deploy!"
   - L'application sera disponible à une URL comme: `https://votre-app.streamlit.app`

### Option 2: Via Git et CLI

```bash
# 1. Ajouter les modifications
git add .

# 2. Commit
git commit -m "Add Streamlit deployment configuration"

# 3. Push vers GitHub
git push origin main

# 4. Déployer via l'interface web (voir Option 1, étape 2)
```

---

## 🔧 Configuration requise

### Fichiers de configuration déjà présents:

✅ **app.py** - Application Streamlit principale
✅ **requirements.txt** - Dépendances Python
✅ **.streamlit/config.toml** - Configuration Streamlit
✅ **outputs_advanced/** - Modèle pré-entraîné et statistiques

### Structure attendue:

```
votre-repo/
├── app.py                          # Application Streamlit
├── requirements.txt                # Dépendances Python
├── README.md                       # Documentation
├── .streamlit/
│   └── config.toml                 # Configuration UI
├── outputs_advanced/
│   ├── best_model.keras            # Modèle entraîné
│   ├── normalization_stats.npz     # Statistiques de normalisation
│   └── test_metrics.json           # Métriques (optionnel)
└── data/                           # Dataset (optionnel pour démo)
    ├── اختر/
    ├── اسفل/
    └── ...
```

---

## ⚙️ Variables d'environnement (si nécessaire)

Si vous avez besoin de secrets ou de variables d'environnement:

1. Dans Streamlit Cloud, allez dans **App settings** > **Secrets**
2. Ajoutez vos secrets au format TOML:

```toml
# Exemple (si nécessaire)
MODEL_PATH = "outputs_advanced/best_model.keras"
```

---

## 📊 Ressources et limites

### Limites Streamlit Cloud (Free Tier):

- **RAM**: 1 GB
- **CPU**: Partagé
- **Storage**: Limité (gardez seulement les fichiers essentiels)
- **Sleep mode**: L'app s'endort après 7 jours d'inactivité

### Optimisations recommandées:

1. **Model caching**: ✅ Déjà implémenté avec `@st.cache_resource`
2. **Data loading**: ✅ Chargement paresseux des données
3. **Fichiers lourds**: Gardez seulement le modèle entraîné et quelques exemples

---

## 🚀 Après le déploiement

### Votre application sera accessible à:

```
https://[votre-nom-app].streamlit.app
```

### Fonctionnalités disponibles:

- ✅ Upload de fichiers EEG CSV
- ✅ Classification en temps réel
- ✅ Visualisations interactives
- ✅ Analyse de samples

### Partage:

- Partagez simplement l'URL avec vos utilisateurs
- Aucune installation requise pour les utilisateurs
- Fonctionne sur desktop et mobile

---

## 🔍 Dépannage

### Erreur "ModuleNotFoundError"
- Vérifiez que toutes les dépendances sont dans `requirements.txt`
- Assurez-vous que les versions sont compatibles

### Erreur "File not found"
- Vérifiez que `outputs_advanced/best_model.keras` est bien commité
- Les chemins doivent être relatifs à la racine du projet

### Application lente
- Le premier chargement peut prendre 30-60 secondes
- Le modèle est mis en cache après le premier chargement
- Considérez réduire la taille du modèle si nécessaire

### Dépassement de mémoire
- Streamlit Cloud Free a 1 GB de RAM
- Si le modèle est trop lourd, considérez:
  - Quantization du modèle
  - Utiliser un plan payant de Streamlit Cloud
  - Déployer sur Heroku/AWS/Azure

---

## 📞 Support

- **Streamlit Docs**: https://docs.streamlit.io/streamlit-cloud
- **Community Forum**: https://discuss.streamlit.io
- **Status Page**: https://status.streamlit.io

---

## 🔄 Mise à jour de l'application

Pour mettre à jour votre application déployée:

```bash
# 1. Faites vos modifications localement
# 2. Testez localement
streamlit run app.py

# 3. Commit et push
git add .
git commit -m "Update: description de vos changements"
git push origin main

# 4. Streamlit Cloud redéploiera automatiquement!
```

L'application se redéploie automatiquement à chaque push sur la branche `main`.

---

## 🎯 Commandes utiles

```bash
# Tester localement
streamlit run app.py

# Vérifier les dépendances
pip list

# Nettoyer le cache
streamlit cache clear

# Voir les logs
# (via l'interface web Streamlit Cloud)
```

---

**Bonne chance avec votre déploiement! 🚀**
