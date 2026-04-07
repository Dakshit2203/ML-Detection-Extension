"""
Tower A - Phase 4 Models - SGD Classifier (Character N-gram Baseline)

SGD-trained linear classifier for the character n-gram TF-IDF feature regime.

This model serves as the n-gram baseline in Phase 4. Rather than using handcrafted features (Groups A–E), it operates
directly on the raw URL string via a TF-IDF character 3–5-gram representation (built by build_tfidf_vectorizer in
utils/features.py). The purpose is to quantify how much of the classification signal can be captured without
domain-specific feature engineering.

SGDClassifier with log_loss loss is equivalent to logistic regression trained with stochastic gradient descent, which
scales well to the sparse high-dimensional feature space produced by TF-IDF (up to 50,000 features). A dense solver
like lbfgs would be impractical at this dimensionality.

alpha=1e-4 is the default L2 regularisation strength. max_iter=1000 is set higher than the sklearn default (1000) to
ensure convergence on the full training vocabulary. tol=1e-4 controls the stopping criterion.

This model is only valid with regime=ngrams in evaluate.py. Passing it a dense AE feature matrix would produce
meaningless results.

Called by evaluate.py via:
    from models.sgd_ngrams import build_model
    model = build_model(seed=42)
"""


from __future__ import annotations

from sklearn.linear_model import SGDClassifier


def build_model(*, seed: int = 42) -> SGDClassifier:
    """
    Return an unfitted SGDClassifier configured for the n-gram TF-IDF regime.

    Parameters
    seed - Random state for reproducibility. Must match the seed passed to set_seeds() in evaluate.py so that results
    are deterministic across runs.
    """
    return SGDClassifier(
        loss="log_loss",    # produces calibrated probabilities via predict_proba
        alpha=1e-4,         # L2 regularisation strength
        max_iter=2000,
        tol=1e-4,
        random_state=seed,
    )
