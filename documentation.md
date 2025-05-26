# Analyse du Risque Bancaire avec Scores ESG

## Aperçu du Projet
Ce projet évalue spécifiquement le risque de défaut des banques en combinant :
- Données financières réelles
- Scores ESG (Environnement, Social, Gouvernance)
- Un modèle mathématique éprouvé

## Comment ça marche ?

### Ce que nous mesurons
1. **Distance au Défaut** : 
   - Indique à quel point une entreprise est proche du défaut de paiement
   - Plus le score est élevé, plus l'entreprise est stable

2. **Probabilité de Défaut** :
   - Traduit la Distance au Défaut en pourcentage de risque
   - De 0% (sans risque) à 100% (défaut certain)

3. **Scores ESG** :
   - Évaluation Environnementale
   - Performance Sociale
   - Qualité de Gouvernance

### Analyse Spécifique aux Banques
- **Score > 3** : Banque très stable (ex: grandes banques systémiques)
- **Score 1-3** : Banque à surveiller (risque modéré)
- **Score ≤ 1** : Banque en difficulté (nécessite attention immédiate)

Les scores ESG révèlent :
- L'impact des pratiques durables sur la stabilité bancaire
- Les points forts/faibles (E, S ou G) spécifiques au secteur bancaire
- La résilience face aux risques climatiques et sociaux

## Données Nécessaires

### Fichier d'Entrée
Un fichier Excel contenant :
- Les informations financières de base :
  - Valeur des actifs
  - Montant de la dette
  - Cours de l'action
- Les scores ESG :
  - Performance environnementale
  - Impact social
  - Qualité de gouvernance

## Résultats Obtenus

Le programme génère un fichier Excel avec :
1. Les données financières originales
2. Les scores de risque calculés :
   - Distance au défaut
   - Probabilité de défaut
3. Les scores ESG
4. Une analyse combinée risque/ESG

## Comment Utiliser les Résultats

### Interprétation Simple
- Entreprise sûre : Score vert (>3)
- Risque modéré : Score orange (1-3) 
- Risque élevé : Score rouge (<1)
