"""
build_dataset.py — Construction du dataset final multi-tables.

Assemble les donnees de six sources (application, bureau, previous_application,
POS/Cash, installments, credit_card_balance) en un seul DataFrame par client.
Appeler build_dataset() pour obtenir le dataset pret pour la modelisation.
"""
import gc

from src.feature_engineering import (
    application_train_test,
    bureau_and_balance,
    previous_applications,
    pos_cash,
    installments_payments,
    credit_card_balance,
)


def build_dataset(num_rows=None):
    """
    Construit le dataset final au niveau client (SK_ID_CURR).

    Combine les données principales avec les historiques bureau,
    anciennes demandes, POS/Cash, échéances et cartes de crédit.
    """  

    # Données principales (train)
    df = application_train_test(num_rows)

    # Ajout des différentes sources de données agrégées
    bureau = bureau_and_balance(num_rows)
    # Jointure avec le dataset principal
    df = df.join(bureau, how="left", on="SK_ID_CURR")
    # Libère la mémoire
    del bureau
    gc.collect()

    # Ajout des anciennes demandes de crédit
    # Agrège l'historique des demandes
    prev = previous_applications(num_rows)
    df = df.join(prev, how="left", on="SK_ID_CURR")
    del prev
    gc.collect()

    # Ajout des données POS/Cash (crédits court terme)
    # Agrège les paiements POS/Cash
    pos = pos_cash(num_rows)
    df = df.join(pos, how="left", on="SK_ID_CURR")
    del pos
    gc.collect()

    # Ajout des données de remboursement (échéances)
    ins = installments_payments(num_rows)
    df = df.join(ins, how="left", on="SK_ID_CURR")
    del ins
    gc.collect()

    # Ajout des données cartes de crédit
    cc = credit_card_balance(num_rows)
    df = df.join(cc, how="left", on="SK_ID_CURR")
    del cc
    gc.collect()

    # Dataset final enrichi
    return df