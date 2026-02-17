"""
fine_tune_llama_lora_ULTRA_OPTIMIZED.py
Version ULTRA-OPTIMISÉE pour:
- GPU GTX 1660 Ti (6GB VRAM)
- RAM limitée (3-4GB disponible)
- Sauvegarde sur E:\ (plus d'espace)
"""

import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from sklearn.model_selection import train_test_split
import json
import gc

# ========================================
# CONFIGURATION ULTRA-OPTIMISÉE
# ========================================
CONFIG = {
    # Modèle le plus léger possible
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    # Dataset
    "dataset_path": "archive/SC_Vuln_8label.csv",
    "test_size": 0.2,
    "random_state": 42,
    "max_samples": 800,  # Limité pour économiser RAM
    
    # LoRA - MINIMAL
    "lora_r": 8,               # Très réduit
    "lora_alpha": 16,          # Très réduit
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],  # Seulement 2 modules
    
    # Training - ULTRA OPTIMISÉ
    "output_dir": r"E:\2-Projet_py\PPE_LoRa_trained",  # ✅ Sauvegarde sur E:\
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,      # ⚠️ BATCH = 1 (minimal)
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,     # Compense le petit batch
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Logging - ESPACÉ
    "logging_steps": 50,
    "save_steps": 200,       # Sauvegardes espacées
    "eval_steps": 200,
    
    # Autres - MINIMAL
    "max_length": 512,       # ⚠️ TRÈS RÉDUIT (au lieu de 2048)
    "use_4bit": True,
    "gradient_checkpointing": True,  # ✅ Économise VRAM
}

LABELS_8 = {
    0: "Block number dependency (BN)",
    1: "Dangerous delegatecall (DE)",
    2: "Ether frozen (EF)",
    3: "Ether strict equality (SE)",
    4: "Integer overflow (OF)",
    5: "Reentrancy (RE)",
    6: "Timestamp dependency (TP)",
    7: "Unchecked external call (UC)",
    8: "Normal"
}


def free_memory():
    """Libère agressivement la mémoire."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_prompt(code, label=None):
    """Crée un prompt COURT pour économiser les tokens."""
    # Version courte pour économiser de la mémoire
    prompt = f"""Analyse ce contrat Solidity et identifie sa vulnérabilité.
Réponds UNIQUEMENT avec un chiffre entre 0 et 8.

Contrat:
{code[:1000]}

Réponse:"""  # Tronque le code à 1000 chars max
    
    if label is not None:
        prompt += f" {label}"
    
    return prompt


def load_and_prepare_data(csv_path, test_size=0.2, random_state=42, max_samples=None):
    """Charge et prépare le dataset (version ultra-optimisée)."""
    print(f"\n📁 Chargement du dataset : {csv_path}")
    
    # Charger avec chunking pour économiser RAM
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    print(f"   Nombre total de contrats : {len(df)}")
    
    # Nettoyer
    df = df.dropna(subset=['code', 'label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    df = df[df['label_encoded'].isin(range(9))]
    
    # ⚠️ LIMITER drastiquement pour économiser RAM
    if max_samples and len(df) > max_samples:
        print(f"\n   ⚠️  LIMITATION à {max_samples} contrats (économie RAM)")
        # Échantillonnage stratifié pour garder la distribution
        df = df.groupby('label_encoded', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // 9), random_state=random_state)
        ).reset_index(drop=True)
    
    print(f"   Après nettoyage : {len(df)} contrats")
    
    # Distribution
    print(f"\n   Distribution des labels :")
    for label, count in df['label_encoded'].value_counts().sort_index().items():
        print(f"      {label} ({LABELS_8.get(label, 'Inconnu')}): {count}")
    
    # Créer les prompts
    print("\n   Création des prompts...")
    df['text'] = df.apply(
        lambda row: create_prompt(row['code'], row['label_encoded']),
        axis=1
    )
    
    # Libérer mémoire
    del df['code']
    free_memory()
    
    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label_encoded']
    )
    
    print(f"\n   Train : {len(train_df)} | Test : {len(test_df)}")
    
    train_dataset = Dataset.from_pandas(train_df[['text']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    # Libérer
    del df, train_df
    free_memory()
    
    return train_dataset, test_dataset, test_df


def load_model_and_tokenizer(model_name, use_4bit=True):
    """Charge le modèle (version ultra-optimisée)."""
    print(f"\n🤖 Chargement du modèle : {model_name}")
    
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,        # ✅ Économise RAM
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("   ✅ Modèle chargé")
    
    free_memory()
    
    return model, tokenizer


def setup_lora(model, config):
    """Configure LoRA (version minimale)."""
    print("\n⚙️  Configuration de LoRA...")
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"   Entraînables : {trainable:,} ({100*trainable/total:.2f}%)")
    
    free_memory()
    
    return model


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize (version économe)."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def train_model(model, tokenizer, train_dataset, test_dataset, config):
    """Entraîne le modèle (version ultra-optimisée)."""
    print("\n🏋️  Début de l'entraînement...")
    
    # Tokenizer avec batch processing
    print("   Tokenization...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        batch_size=10,  # Petit batch pour tokenization
        remove_columns=train_dataset.column_names,
        desc="Train"
    )
    
    free_memory()
    
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        batch_size=10,
        remove_columns=test_dataset.column_names,
        desc="Test"
    )
    
    free_memory()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Arguments ULTRA-OPTIMISÉS
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        save_total_limit=1,              # ✅ Garde seulement 1 checkpoint
        gradient_checkpointing=True,     # ✅ Économise VRAM
        optim="paged_adamw_8bit",        # ✅ Optimiseur 8-bit
        dataloader_num_workers=0,        # ✅ Pas de workers parallèles
        dataloader_pin_memory=False,     # ✅ Économise RAM
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    print("\n   🚀 Lancement...")
    print(f"   ⏱️  Durée estimée : 2-4 heures")
    print(f"   💾 Sauvegarde sur : {config['output_dir']}")
    print()
    
    # Entraîner
    trainer.train()
    
    print("\n   ✅ Terminé !")
    
    # Sauvegarder
    final_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"   💾 Modèle : {final_path}")
    
    free_memory()
    
    return trainer


def evaluate_model(model, tokenizer, test_df, max_length=512, max_samples=50):
    """Évalue (version économe)."""
    print(f"\n📊 Évaluation sur {max_samples} échantillons...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    test_sample = test_df.sample(n=min(max_samples, len(test_df)))
    
    for idx, row in test_sample.iterrows():
        prompt = create_prompt(row['code'])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        predicted_label = -1
        for char in response:
            if char.isdigit() and int(char) in range(9):
                predicted_label = int(char)
                break
        
        predictions.append(predicted_label)
        true_labels.append(row['label_encoded'])
        
        if len(predictions) % 10 == 0:
            free_memory()
    
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions) * 100
    
    print(f"\n   ✅ Précision : {accuracy:.2f}% ({correct}/{len(predictions)})")
    
    return predictions, true_labels, accuracy


def save_config(config, output_dir):
    """Sauvegarde la config."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Config : {config_path}")


# ========================================
# PROGRAMME PRINCIPAL
# ========================================
if __name__ == "__main__":
    print("="*70)
    print("FINE-TUNING LLAMA - VERSION ULTRA-OPTIMISÉE")
    print("="*70)
    print(f"\n⚙️  Configuration :")
    print(f"   GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Modèle : {CONFIG['model_name']}")
    print(f"   Samples : {CONFIG['max_samples']}")
    print(f"   Batch size : {CONFIG['per_device_train_batch_size']}")
    print(f"   Max length : {CONFIG['max_length']}")
    print(f"   Sauvegarde : {CONFIG['output_dir']}")
    print("="*70)
    
    # Vérifier le disque E:\
    if not os.path.exists("E:\\"):
        print("\n⚠️  ATTENTION : Le disque E:\ n'existe pas !")
        print("   Modification de la sauvegarde vers C:\\")
        CONFIG["output_dir"] = r"C:\0-Code_py_temp\Projet_PPE\PPE_LoRa_trained"
    
    # Créer le dossier de sortie
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"\n✅ Dossier de sortie créé : {CONFIG['output_dir']}")
    
    response = input("\nContinuer ? (y/n): ")
    if response.lower() != 'y':
        print("Annulé")
        exit(0)
    
    # 1. Charger les données
    train_dataset, test_dataset, test_df = load_and_prepare_data(
        CONFIG["dataset_path"],
        CONFIG["test_size"],
        CONFIG["random_state"],
        CONFIG["max_samples"]
    )
    
    # 2. Charger le modèle
    model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"],
        CONFIG["use_4bit"]
    )
    
    # 3. Setup LoRA
    model = setup_lora(model, CONFIG)
    
    # 4. Sauvegarder config
    save_config(CONFIG, CONFIG["output_dir"])
    
    # 5. Entraîner
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, CONFIG)
    
    # 6. Évaluer
    predictions, true_labels, accuracy = evaluate_model(
        model, tokenizer, test_df, CONFIG["max_length"], max_samples=50
    )
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ !")
    print("="*70)
    print(f"\n📁 Modèle : {CONFIG['output_dir']}/final_model")
    print(f"📊 Précision : {accuracy:.2f}%")
    print(f"💾 Espace utilisé : ~2-3GB sur E:\\")
