# Chargement du service
# MODEL_PATH = "/content/mba_service.pkl"
# service_loaded = MBAService.load(MODEL_PATH)

# Produit a analyser
produit_recherche = "Al Muhafiz Sohan Halwa Almond"

# Recommandations avec FP-Growth
recommandations = service_loaded.get_associated_products(
    produit_recherche, method="fpgrowth", top_n=5
)

print(f"Recommandations pour '{produit_recherche}' avec FP-Growth :")
if recommandations:
    for i, r in enumerate(recommandations, 1):
        print(f"{i}. {r['consequent']} | conf={r['confidence']:.3f} | supp={r['support']:.4f} | lift={r['lift']:.3f}")
else:
    print("Aucune recommandation trouvee.")
