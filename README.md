# Fine-tuning de LLaMA avec LoRA pour la Classification de Vulnérabilités Solidity

Ce projet permet de fine-tuner le modèle LLaMA avec la technique LoRA (Low-Rank Adaptation) pour classifier les vulnérabilités dans les smart contracts Solidity.

## 📋 Prérequis

### Matériel
- **GPU recommandé** : NVIDIA GPU avec au moins 8GB de VRAM (16GB+ idéal)
- **RAM** : Au moins 16GB
- **Espace disque** : ~20GB pour le modèle et les données

### Logiciels
- Python 3.8+
- CUDA 11.8+ (pour l'utilisation GPU)
- pip

## 🚀 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Vérifier l'installation de PyTorch avec CUDA

```bash
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 📊 Préparation des données

Assurez-vous d'avoir le dataset `SC_Vuln_8label.csv` dans le dossier `archive/` avec la structure suivante :
- `filename` : nom du fichier
- `code` : code Solidity du contrat
- `label_encoded` : label de la vulnérabilité (0-8)

Les 9 classes de vulnérabilités :
- 0: Block number dependency (BN)
- 1: Dangerous delegatecall (DE)
- 2: Ether frozen (EF)
- 3: Ether strict equality (SE)
- 4: Integer overflow (OF)
- 5: Reentrancy (RE)
- 6: Timestamp dependency (TP)
- 7: Unchecked external call (UC)
- 8: Normal (sans vulnérabilité)

## 🏋️ Entraînement du modèle

### Configuration de base

Modifiez les paramètres dans `fine_tune_llama_lora.py` si nécessaire :

```python
CONFIG = {
    "model_name": "meta-llama/Llama-3.2-3B",  # Modèle de base
    "num_train_epochs": 3,                     # Nombre d'époques
    "lora_r": 16,                              # Rang LoRA
    "learning_rate": 2e-4,                     # Taux d'apprentissage
    # ... autres paramètres
}
```

### Lancer l'entraînement

```bash
python fine_tune_llama_lora.py
```

**Important** : Si vous n'avez pas accès au modèle LLaMA officiel, vous pouvez utiliser des alternatives open-source :
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (plus léger, ~1GB)
- `microsoft/phi-2` (2.7B paramètres)

Modifiez simplement `CONFIG["model_name"]` dans le script.

### Durée estimée

- Avec GPU RTX 3090 (24GB) : ~2-4 heures pour 3 époques
- Avec GPU RTX 3060 (12GB) : ~4-8 heures
- Avec CPU (non recommandé) : plusieurs jours

### Réduction de la mémoire

Si vous manquez de VRAM, réduisez :
```python
"per_device_train_batch_size": 2,  # au lieu de 4
"gradient_accumulation_steps": 8,  # au lieu de 4
"max_length": 1024,                # au lieu de 2048
```

## 🔮 Utilisation du modèle fine-tuné

### Mode batch (classifier un CSV entier)

```bash
python inference_lora.py archive/SC_Vuln_8label.csv resultats.csv
```

### Mode interactif

```bash
python inference_lora.py
```

Puis collez votre code Solidity et tapez `END` pour obtenir la prédiction.

### Exemple d'utilisation en Python

```python
from inference_lora import load_finetuned_model, classify_contract

# Charger le modèle
model, tokenizer = load_finetuned_model(
    "meta-llama/Llama-3.2-3B",
    "./llama_lora_solidity_finetuned/final_model"
)

# Classifier un contrat
code_solidity = """
pragma solidity ^0.4.0;
contract MyContract {
    function withdraw() public {
        msg.sender.call.value(balance)();
        balance = 0;
    }
}
"""

prediction = classify_contract(model, tokenizer, code_solidity)
print(f"Vulnérabilité détectée : {prediction}")
```

## 📈 Résultats attendus

Après le fine-tuning, vous devriez obtenir :
- **Précision baseline (LLaMA non fine-tuné)** : ~30-40%
- **Précision après fine-tuning** : ~70-85%+ (selon le dataset et les hyperparamètres)

Les résultats sont sauvegardés dans `llama_lora_solidity_finetuned/` :
- `final_model/` : poids LoRA du modèle
- `predictions.csv` : prédictions sur le test set
- `training_config.json` : configuration de l'entraînement

## ⚙️ Optimisations avancées

### 1. Ajuster les hyperparamètres LoRA

```python
"lora_r": 32,        # Augmenter pour plus de capacité (mais plus de mémoire)
"lora_alpha": 64,    # Double de lora_r généralement
"lora_dropout": 0.1, # Augmenter si overfitting
```

### 2. Modules cibles supplémentaires

```python
"target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"  # Pour MLP aussi
]
```

### 3. Learning rate scheduling

```python
"lr_scheduler_type": "cosine",      # ou "linear", "polynomial"
"warmup_ratio": 0.05,               # 5% des steps en warmup
```

### 4. Data augmentation

Vous pouvez augmenter artificiellement le dataset en :
- Reformulant les prompts
- Ajoutant du bruit contrôlé au code
- Utilisant des variations de formatting

## 🐛 Dépannage

### Erreur CUDA Out of Memory
```python
# Réduire la taille des batchs
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 16,

# Ou utiliser une quantification plus agressive
"use_4bit": True,
```

### Modèle LLaMA non accessible
Utilisez une alternative open-source :
```python
"model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Entraînement très lent
- Vérifiez que vous utilisez bien le GPU : `torch.cuda.is_available()`
- Réduisez `max_length` à 1024 ou 512
- Augmentez `gradient_accumulation_steps`

### Précision faible après entraînement
- Augmentez le nombre d'époques
- Ajustez le learning rate (essayez 1e-4 ou 3e-4)
- Vérifiez la distribution des classes (déséquilibre ?)

## 📝 Structure des fichiers

```
.
├── fine_tune_llama_lora.py      # Script d'entraînement principal
├── inference_lora.py            # Script d'inférence
├── requirements.txt             # Dépendances
├── README.md                    # Ce fichier
├── archive/
│   └── SC_Vuln_8label.csv      # Dataset d'entraînement
└── llama_lora_solidity_finetuned/
    ├── final_model/            # Modèle fine-tuné
    ├── predictions.csv         # Résultats
    └── training_config.json    # Configuration
```

## 🔍 Monitoring de l'entraînement

Pour suivre l'entraînement en temps réel, installez TensorBoard :

```bash
pip install tensorboard
tensorboard --logdir=llama_lora_solidity_finetuned
```

## 📚 Ressources supplémentaires

- [Documentation LoRA](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [LLaMA](https://ai.meta.com/llama/)

## 💡 Astuces

1. **Commencez petit** : Testez d'abord avec un petit subset (100-500 contrats) pour valider le pipeline
2. **Sauvegardez souvent** : Utilisez `save_steps=100` pour ne pas perdre de progrès
3. **Monitorer la loss** : Si elle ne descend pas, ajustez le learning rate
4. **Test early stopping** : Si la validation loss augmente, arrêtez l'entraînement

## ⚖️ Licence

Ce code est fourni à des fins éducatives et de recherche.

## 🤝 Contribution

Les améliorations sont les bienvenues ! N'hésitez pas à ouvrir des issues ou des pull requests.
