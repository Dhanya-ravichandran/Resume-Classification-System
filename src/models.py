from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def get_models():
    """
    Returns a dictionary of machine learning models
    for resume classification.
    """

    models = {
        "Naive_Bayes": MultinomialNB(),

        "Logistic_Regression": LogisticRegression(
            max_iter=2000,
            C=2.0,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": LinearSVC(
            C=1.5,
            class_weight="balanced",
            random_state=42
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight="balanced",
            random_state=42
        ),

        "Decision_Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "ANN": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=42
        )
    }

    return models