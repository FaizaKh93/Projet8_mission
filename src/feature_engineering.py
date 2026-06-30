"""
feature_engineering.py — Pipeline de feature engineering.

Nettoie, encode et agregee chaque table source du dataset Home Credit :
application, bureau, previous_application, POS/Cash, installments, credit_card_balance.
Chaque fonction retourne un DataFrame indexe par SK_ID_CURR, pret a etre joint
au dataset principal via build_dataset.build_dataset().
"""
import gc
import numpy as np
import pandas as pd

from src.data_loading import load_csv


# ── Encodage one-hot ───────────────────────────────────────────────────────────
def one_hot_encoder(df, nan_as_category=True):
    """
    Applique un encodage one-hot sur toutes les colonnes 'object' du DataFrame.

    Retour
    ------
    df : pd.DataFrame
        DataFrame avec les colonnes categorielless remplacees par des colonnes binaires.
    new_columns : list[str]
        Noms des colonnes creees par le one-hot encoding.
    categorical_columns : list[str]
        Noms des colonnes originales encodees.
    """
    # Sauvegarde des colonnes initiales
    original_columns = list(df.columns)

    # Sélection des colonnes catégorielles (type object)
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    #categorical_columns = [col for col in df.columns if df[col].dtype == ["object", "category"]]

    # Application du one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)

    # Identification des nouvelles colonnes créées
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns, categorical_columns

#=================================================================
# Preprocess application_train.csv and application_test.csv
#=================================================================
def application_train_test(num_rows=None, nan_as_category=False):
    """
    Charge, nettoie et enrichit les données des fichiers application_train (ou application_test).

    Cette fonction :
    - Nettoie les anomalies, encode les variables catégorielles 
      et crée des ratios financiers et scores externes agrégés.
    """

    # Chargement des données train et test
    df = load_csv("application_train.csv", nrows=num_rows)
    # test_df = load_csv("application_test.csv", nrows=num_rows)

    #print(f"Train samples: {len(df)}, test samples: {len(test_df)}")
    print(f"Train samples: {len(df)}")

    # Fusion des datasets pour traitement uniforme
    #df = pd.concat([df, test_df], axis=0).reset_index(drop=True)

    # Suppression des valeurs aberrantes (genre inconnu 'XNA')
    df = df[df["CODE_GENDER"] != "XNA"]

    # Encodage des variables binaires (2 modalités)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    # Encodage des variables catégorielles restantes (one-hot)
    df, cat_cols, original_cat_cols = one_hot_encoder(df, nan_as_category)

    # créer un flag pour signaler l'anomalie (365243 = valeur manquante déguisée)
    df['DAYS_EMPLOYED_ANOM'] = (df["DAYS_EMPLOYED"] == 365243).astype(int)

    # Correction des anomalies avec une valeur aberrante connue 
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan) 

    #------------------------------
    # Feature engineering
    #------------------------------
    # Ratio ancienneté emploi / âge
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # Ratio revenu / crédit (capacité de remboursement)
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]

    # Revenu par membre du foyer
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # Ratio annuité / revenu (pression financière)
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # Taux de remboursement du crédit
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # moyenne des EXT_SOURCE_x 
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

   # variance des EXT_SOURCE_x 
    df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)

    # Libération mémoire
    #del test_df
    #gc.collect()

    return df

#=================================================================
# Preprocess bureau.csv and bureau_balance.csv
#=================================================================
def bureau_and_balance(num_rows = None, nan_as_category = True):
    """
    Agrège les historiques de crédits externes au niveau client.

    Combine bureau.csv avec bureau_balance.csv, crée des ratios de dette
    et agrège les crédits globaux, actifs et clôturés.

    Notes
    -----
    - bureau_balance contient un historique mensuel → agrégé par SK_ID_BUREAU
    - bureau contient les crédits → agrégé par SK_ID_CURR
    - Les variables sont fortement enrichies via des statistiques (mean, max, sum...)
    """
    #------------------------------
    # Chargement des données
    #------------------------------   
    bureau = load_csv('bureau.csv', nrows = num_rows)
    bb = load_csv('bureau_balance.csv', nrows = num_rows)

    # Encodage des variables catégorielles
    bb, bb_cat, original_bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat, original_bureau_cat = one_hot_encoder(bureau, nan_as_category)

    #------------------------------
    # Agrégation bureau_balance (niveau crédit). Bureau balance: Perform aggregations and merge with bureau.csv
    #------------------------------     
    # Statistiques de base sur l'historique mensuel
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    # Moyenne des catégories encodées
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    
    # Agrégation par crédit (SK_ID_BUREAU)    
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

    # Renommage des colonnes
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    # Jointure avec bureau
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
   
    # Suppression clé inutile après jointure
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # ratio dette / crédit : grand ratio --> dette encore lourde & forte exposition fiancière
    bureau["DEBT_PERCENTAGE"] = (bureau["AMT_CREDIT_SUM_DEBT"] / bureau["AMT_CREDIT_SUM"])

    # crédit en retard relatif : grand ratio --> beaucoup de retard
    bureau["OVERDUE_PERCENTAGE"] = (bureau["AMT_CREDIT_SUM_OVERDUE"] / bureau["AMT_CREDIT_SUM"])

    #------------------------------
    # Définition des agrégations
    #------------------------------
    # Variables numériques
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DEBT_PERCENTAGE': ['mean', 'max'],
        'OVERDUE_PERCENTAGE': ['mean', 'max'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Variables catégorielles (one-hot → moyennes)
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    #------------------------------
    # Agrégation globale au niveau client
    #------------------------------
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
   
    # Préfixe pour identifier les features
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
 
    #------------------------------
    # Features spécifiques : crédits actifs
    #------------------------------
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    #------------------------------
    # Features spécifiques : crédits clôturés
    #------------------------------
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


#=================================================================
# Preprocess previous_applications.csv
#=================================================================
def previous_applications(num_rows = None, nan_as_category = True):
    """
    Agrège les anciennes demandes de crédit au niveau client.

    Crée des ratios d’acceptation et d’apport, puis résume
    les demandes globales, approuvées et refusées.
    """

    # Chargement des anciennes demandes de crédit
    prev = load_csv('previous_application.csv', nrows = num_rows)

    # Encodage des variables catégorielles
    prev, cat_cols, original_cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan

    # Remplacement des valeurs aberrantes 365243 par NaN
    # Dans ce dataset, 365243 représente souvent une date manquante déguisée.
    date_cols = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
    ] 

    prev[date_cols] = prev[date_cols].replace(365243, np.nan)

    #prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    #prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    #prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    #prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    #prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    #------------------------------
    # Features engineering
    #------------------------------
    # ratio entre le montant demandé par le client et le montant réellement accordé.
    prev['CREDIT_APP_PERC'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION'] 

    # ratio entre l'apport et le crédit accordé
    prev["DOWN_PAYMENT_CREDIT_RATIO"] = (prev["AMT_DOWN_PAYMENT"] / prev["AMT_CREDIT"])

    # Agrégations numériques des anciennes demandes
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_APP_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'DOWN_PAYMENT_CREDIT_RATIO': ['min', 'max', 'mean']
    }

    # Agrégations des variables catégorielles encodées
    # La moyenne d'une colonne one-hot représente la proportion de cette modalité.
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    # Agrégation globale au niveau client
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    # Renommage des colonnes avec le préfixe PREV_
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Agrégation spécifique des demandes approuvées
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Agrégation spécifique des demandes refusées
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    # Libération mémoire
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

#=================================================================
# Preprocess POS_CASH_balance.csv
#=================================================================
def pos_cash(num_rows = None, nan_as_category = True):
    """
    Agrège l’historique mensuel POS/Cash au niveau client.

    Résume les retards, les statuts contractuels et le volume
    d’historique disponible.
    """
    # Chargement de l’historique mensuel POS/Cash
    pos = load_csv('POS_CASH_balance.csv', nrows = num_rows)

    # Encodage des variables catégorielles
    pos, cat_cols, original_cat_cols = one_hot_encoder(pos, nan_as_category= True)

    # Définition des agrégations numériques principales
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }

    # Agrégation des variables catégorielles encodées
    # La moyenne représente la proportion de chaque modalité pour un client.
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    # Agrégation au niveau client
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)

    # Renommage des colonnes avec le préfixe POS_
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Nombre total de lignes POS/Cash associées à chaque client
    # Cela donne une mesure du volume d’historique disponible.
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    
    # Libération mémoire
    del pos
    gc.collect()
    return pos_agg

#=================================================================
# Preprocess installments_payments.csv
#=================================================================
def installments_payments(num_rows = None, nan_as_category = True):
    """
    Agrège les paiements d’échéances au niveau client.

    Crée des indicateurs de retard, paiement anticipé,
    respect du montant dû et fréquence des retards.
    -----
    Notes
    -----
    - Ce dataset est l’un des plus importants pour prédire le défaut.
    - Il reflète directement le comportement réel de remboursement du client.
    """
    #------------------------------
    # Chargement des données
    #------------------------------
    ins = load_csv('installments_payments.csv', nrows = num_rows)

    # Encodage des variables catégorielles
    ins, cat_cols, original_cat_cols = one_hot_encoder(ins, nan_as_category= True)

    #------------------------------
    # feature engineering
    #------------------------------   
    # Ratio paiement / montant attendu (respect des échéances)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']

    # Différence entre montant attendu et payé
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Retard de paiement (Days Past Due)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']


    # Paiement anticipé (Days Before Due)
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']

    # On ne garde que les valeurs positives
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # flag de retard
    ins["LATE_PAYMENT_FLAG"] = (ins["DPD"] > 0).astype(int)

    #------------------------------
    # aggregations
    #------------------------------
    aggregations = {
        # Nombre de versions d’échéances (modifications du plan)
        'NUM_INSTALMENT_VERSION': ['nunique'],
        # Retards de paiement
        'DPD': ['max', 'mean', 'sum'],
        # Paiements anticipés
        'DBD': ['max', 'mean', 'sum'],
        # Respect des paiements
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        # Différence paiement attendu vs réel
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        # Montants
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        # flage de retard
        'LATE_PAYMENT_FLAG': ['mean', 'sum'],
        # Historique des paiements
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    # Agrégations des variables catégorielles encodées
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    # Agrégation au niveau client
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)

    # Renommage des colonnes
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Nombre total d’échéances par client
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    # Libération mémoire
    del ins
    gc.collect()
    return ins_agg

#=================================================================
# Preprocess credit_card_balance.csv
#=================================================================
def credit_card_balance(num_rows = None, nan_as_category = True):
    """
    Agrège l’historique de cartes de crédit au niveau client.

    Crée un ratio d’utilisation de limite et résume les comportements
    de solde, paiement, retrait et retard.
    -----
    Notes
    -----
    - Les données sont initialement au niveau transactionnel (par carte).
    - On agrège tout au niveau client pour la modélisation.
    - Les statistiques (min, max, mean, sum, var) capturent le comportement financier.
    """

    #------------------------------
    # chargement des données
    #------------------------------
    cc = load_csv('credit_card_balance.csv', nrows = num_rows)
    
    # Encodage des variables catégorielles
    cc, cat_cols, original_cat_cols = one_hot_encoder(cc, nan_as_category= True)

    # Suppression de l'identifiant de crédit (on agrège au niveau client)
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    
    # part de la limite de crédit utilisée
    cc["BALANCE_LIMIT_RATIO"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    #------------------------------
    # aggregation globale
    #------------------------------
    # Application de statistiques sur toutes les colonnes numériques
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    
    # Renommage des colonnes avec préfixe CC_
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
 
    #------------------------------
    # Feature supplémentaire
    #------------------------------ 
    # Nombre de lignes liées aux cartes de crédit (volume d’activité)
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    
   
    # Libération mémoire
    del cc
    gc.collect()
    return cc_agg