# NNEinFact
Source code for the paper [Near-Universal Multiplicative Updates for Nonnegative Einsum Factorization](https://arxiv.org/abs/2602.02759) by John Hood and Aaron Schein. 

NNEinFact is a general-purpose library for fitting any nonnegative tensor factorization expressible as an `einsum` contraction. It implements a multiplicative update algorithm guaranteed to converge under a wide family of loss functions, including the $(\alpha, \beta)$-divergence.

<img width="1211" height="663" alt="Screenshot 2026-02-02 at 12 10 36 PM" src="https://github.com/user-attachments/assets/caf2577c-6acb-4172-abc7-b18efdbae7ad" />

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Capabilities
* **Modeling**: Fit CP, Tucker, Tensor-Train, or custom models using simple string notation.
* **Loss Functions**: The current implementation supports any $(\alpha, \beta)$ divergence other than $$\alpha = 0, \beta \neq 1$$. Examples include Euclidean distance $$(\alpha=1.0, \beta =1.0)$$, KL divergence $$(\alpha=1.0, \beta = 0.0)$$, Hellinger distance $$(\alpha=\beta = 0.5)$$, among others. 

For faster training, we recommend using GPUs when possible. If your data $$Y$$ contains exact zeros, use $$\alpha > 0$$.

### Quick Start
```python
from einfact import NNEinFact

Y = np.load(...)

shape_dict = {**dict(zip(model_str.split('->')[-1], Y.shape)), 'k': 6, 'r': 10}
# Define a custom model:
model_str = 'wr,hr,dr,ikr,jkr -> whdij' #custom model

#model_str = 'wr,hr,dr,ir,jr->whdij' #CP
#model_str = 'wa,hb,dc,if,je,abcfe->whdij' #Tucker (add 'a', 'b', 'c', 'f', 'e' to shape_dict)
#model_str = 'wa,hab,dbc,icf,jf->whdij' #tensor-train (add 'a', 'b', 'c', 'f' to shape_dict)

model = NNEinFact(
    model_str,alpha=0.5, beta=0.5,
    shape_dict=shape_dict, device='cuda'
)


# Fit to your data
model.fit(Y)

# Retrieve factors
factors = model.get_params()

 1. L'argument obligatoire :

  • --input-dir <chemin_vers_le_dossier> : Il est indispensable pour spécifier le dossier contenant les fichiers bruts
  ICEWS (.tab ou .tsv).

  2. L'argument à modifier pour coller au papier :

  • --heldout-frac 0.1 : Dans le fichier 6_empirics.txt, les auteurs précisent : "we randomly assign each element of Y to
  the training set with 90% probability and otherwise assign it to a heldout set" (ce qui implique que le heldout set
  représente 10% des données). Or, dans le script prepare_icews.py, la valeur par défaut de --heldout-frac est définie à
  0.05. Il faut donc explicitement la passer à 0.1.

  3. Les arguments optionnels (car leurs valeurs par défaut sont déjà correctes) :

  • --start-year 1995 et --end-year 2013 : Le papier mentionne utiliser les données "spanning 1995--2013". Le script
  utilise par chance déjà ces valeurs par défaut, il n'est donc pas strictement nécessaire de les redéclarer, mais vous
  pouvez le faire par souci d'explicitation.
  • --seed 29482 : Déjà défini par défaut dans le code pour fixer l'aléatoire et garantir la reproductibilité de la
  répartition train/heldout.


