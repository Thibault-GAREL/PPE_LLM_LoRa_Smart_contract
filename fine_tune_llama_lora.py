"""
fine_tune_llama_lora.py
Script de fine-tuning de LLaMA avec LoRA pour la classification de vulnérabilités Solidity
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

# ========================================
# CONFIGURATION
# ========================================
CONFIG = {
    # Modèle de base
    "model_name": "meta-llama/Llama-3.2-3B",  # ou "meta-llama/Llama-3.2-1B" pour plus léger
    
    # Dataset
    "dataset_path": "archive/SC_Vuln_8label.csv",
    "test_size": 0.2,
    "random_state": 42,
    
    # LoRA parameters
    "lora_r": 16,              # Rang de la matrice LoRA (plus petit = moins de paramètres)
    "lora_alpha": 32,          # Facteur de scaling
    "lora_dropout": 0.05,      # Dropout pour régularisation
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Modules à adapter
    
    # Training parameters
    "output_dir": "./llama_lora_solidity_finetuned",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    
    # Logging
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    
    # Autres
    "max_length": 2048,
    "use_4bit": True,  # Quantification 4-bit pour économiser de la mémoire
}

# Labels pour le dataset 8 classes
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


def create_prompt(code, label=None):
    """
    Crée le prompt pour l'entraînement ou l'inférence.
    
    Args:
        code: Code Solidity
        label: Label de classification (optionnel, pour l'entraînement)
    """
    labels_description = "\n".join([
        f"- {i} si le contrat a une vulnérabilité de type \"{desc}\""
        if i < 8 else f"- {i} si le contrat est normal (sans vulnérabilité)"
        for i, desc in LABELS_8.items()
    ])
    
    prompt = f"""Analyse ce contrat Solidity et identifie s'il contient une vulnérabilité.

Réponds UNIQUEMENT avec UN SEUL chiffre entre 0 et 8 :
{labels_description}

IMPORTANT : Réponds UNIQUEMENT avec le chiffre, rien d'autre.

Contrat Solidity à analyser :
{code}

Réponse (un seul chiffre) :"""
    
    if label is not None:
        prompt += f" {label}"
    
    return prompt


def load_and_prepare_data(csv_path, test_size=0.2, random_state=42):
    """
    Charge et prépare le dataset pour l'entraînement.
    """
    print(f"\n📁 Chargement du dataset : {csv_path}")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    print(f"   Nombre total de contrats : {len(df)}")
    
    # Vérifier les colonnes nécessaires
    if 'code' not in df.columns or 'label_encoded' not in df.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'code' et 'label_encoded'")
    
    # Nettoyer les données
    df = df.dropna(subset=['code', 'label_encoded'])
    df['label_encoded'] = df['label_encoded'].astype(int)
    
    print(f"   Après nettoyage : {len(df)} contrats")
    print(f"\n   Distribution des labels :")
    for label, count in df['label_encoded'].value_counts().sort_index().items():
        print(f"      {label} ({LABELS_8[label]}): {count}")
    
    # Créer les prompts
    df['text'] = df.apply(
        lambda row: create_prompt(row['code'], row['label_encoded']),
        axis=1
    )
    
    # Split train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_encoded']
    )
    
    print(f"\n   📊 Split des données :")
    print(f"      Train : {len(train_df)} contrats")
    print(f"      Test  : {len(test_df)} contrats")
    
    # Convertir en Dataset Hugging Face
    train_dataset = Dataset.from_pandas(train_df[['text']])
    test_dataset = Dataset.from_pandas(test_df[['text']])
    
    return train_dataset, test_dataset, test_df


def load_model_and_tokenizer(model_name, use_4bit=True):
    """
    Charge le modèle et le tokenizer avec quantification optionnelle.
    """
    print(f"\n🤖 Chargement du modèle : {model_name}")
    
    # Configurer la quantification 4-bit si demandée
    if use_4bit:
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
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Configurer le padding token si nécessaire
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"   ✅ Modèle chargé avec succès")
    return model, tokenizer


def setup_lora(model, config):
    """
    Configure LoRA sur le modèle.
    """
    print(f"\n⚙️  Configuration de LoRA")
    
    # Préparer le modèle pour l'entraînement avec quantification
    model = prepare_model_for_kbit_training(model)
    
    # Configuration LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Appliquer LoRA
    model = get_peft_model(model, lora_config)
    
    # Afficher les paramètres entraînables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"   Paramètres entraînables : {trainable_params:,}")
    print(f"   Tous les paramètres : {all_params:,}")
    print(f"   Pourcentage entraînable : {100 * trainable_params / all_params:.2f}%")
    
    return model


def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize les exemples pour l'entraînement.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )


def train_model(model, tokenizer, train_dataset, test_dataset, config):
    """
    Entraîne le modèle avec LoRA.
    """
    print(f"\n🏋️  Début de l'entraînement")
    
    # Tokenizer les datasets
    print("   Tokenization des données...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["max_length"]),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Arguments d'entraînement
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
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",  # Désactiver wandb, tensorboard, etc.
        save_total_limit=3,
    )
    
    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
    )
    
    # Entraîner
    print("   🚀 Lancement de l'entraînement...")
    trainer.train()
    
    print(f"\n   ✅ Entraînement terminé !")
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(config["output_dir"], "final_model")
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"   💾 Modèle sauvegardé dans : {final_model_path}")
    
    return trainer


def evaluate_model(model, tokenizer, test_df, max_length=2048):
    """
    Évalue le modèle sur le test set.
    """
    print(f"\n📊 Évaluation du modèle")
    
    predictions = []
    true_labels = []
    
    model.eval()
    
    for idx, row in test_df.iterrows():
        # Créer le prompt sans le label
        prompt = create_prompt(row['code'])
        
        # Tokenizer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Générer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Décoder
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        # Extraire le chiffre prédit
        predicted_label = -1
        for char in response:
            if char.isdigit() and int(char) in range(9):
                predicted_label = int(char)
                break
        
        predictions.append(predicted_label)
        true_labels.append(row['label_encoded'])
        
        if (idx + 1) % 10 == 0:
            print(f"   Progression : {idx + 1}/{len(test_df)}")
    
    # Calculer la précision
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(predictions) * 100
    
    print(f"\n   ✅ Précision sur le test set : {accuracy:.2f}%")
    print(f"   Prédictions correctes : {correct}/{len(predictions)}")
    
    # Distribution des erreurs
    print(f"\n   Distribution des erreurs par classe :")
    for label in range(9):
        label_indices = [i for i, t in enumerate(true_labels) if t == label]
        if label_indices:
            label_correct = sum(predictions[i] == label for i in label_indices)
            label_total = len(label_indices)
            label_acc = label_correct / label_total * 100
            print(f"      {label} ({LABELS_8[label]}): {label_acc:.1f}% ({label_correct}/{label_total})")
    
    return predictions, true_labels, accuracy


def save_config(config, output_dir):
    """
    Sauvegarde la configuration de l'entraînement.
    """
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Configuration sauvegardée dans : {config_path}")


# ========================================
# PROGRAMME PRINCIPAL
# ========================================
if __name__ == "__main__":
    print("="*60)
    print("FINE-TUNING DE LLAMA AVEC LoRA")
    print("Classification de vulnérabilités Solidity")
    print("="*60)
    
    # Vérifier CUDA
    if torch.cuda.is_available():
        print(f"\n✅ GPU disponible : {torch.cuda.get_device_name(0)}")
        print(f"   Mémoire GPU : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  Aucun GPU détecté, l'entraînement sera très lent sur CPU")
        response = input("Continuer quand même ? (y/n): ")
        if response.lower() != 'y':
            exit(0)
    
    # 1. Charger les données
    train_dataset, test_dataset, test_df = load_and_prepare_data(
        CONFIG["dataset_path"],
        CONFIG["test_size"],
        CONFIG["random_state"]
    )
    
    # 2. Charger le modèle et tokenizer
    model, tokenizer = load_model_and_tokenizer(
        CONFIG["model_name"],
        CONFIG["use_4bit"]
    )
    
    # 3. Configurer LoRA
    model = setup_lora(model, CONFIG)
    
    # 4. Sauvegarder la configuration
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    save_config(CONFIG, CONFIG["output_dir"])
    
    # 5. Entraîner
    trainer = train_model(model, tokenizer, train_dataset, test_dataset, CONFIG)
    
    # 6. Évaluer
    predictions, true_labels, accuracy = evaluate_model(model, tokenizer, test_df, CONFIG["max_length"])
    
    # 7. Sauvegarder les résultats
    results_df = test_df.copy()
    results_df['prediction'] = predictions
    results_path = os.path.join(CONFIG["output_dir"], "predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Prédictions sauvegardées dans : {results_path}")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ !")
    print("="*60)
