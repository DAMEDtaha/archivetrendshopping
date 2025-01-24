
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import graphviz

class ID3:
    def fit(self, X, y):
        
        def entropy(y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log2(p) for p in ps if p > 0])

        
        def gain(X, y, attribute):
            values = np.unique(X[:, attribute])
            entropy_values = []
            for value in values:
                y_subset = y[X[:, attribute] == value]
                entropy_values.append(entropy(y_subset))
            return entropy(y) - np.sum([len(y_subset) / len(y) * e for y_subset, e in zip(entropy_values, values)])

        
        def build_tree(X, y, attributes):
            if len(np.unique(y)) == 1:
                return np.unique(y)[0]
            elif len(attributes) == 0:
                return np.bincount(y).argmax()
            else:
                best_attribute = np.argmax([gain(X, y, attribute) for attribute in attributes])
                tree = {attributes[best_attribute]: {}}
                for value in np.unique(X[:, best_attribute]):
                    y_subset = y[X[:, best_attribute] == value]
                    X_subset = X[X[:, best_attribute] == value]
                    tree[attributes[best_attribute]][value] = build_tree(X_subset, y_subset, [a for a in attributes if a != attributes[best_attribute]])
                return tree

        
        attributes = list(range(X.shape[1]))
        self.tree = build_tree(X, y, attributes)

class CART:
    def fit(self, X, y):
        
        self.model = DecisionTreeClassifier(criterion='gini')
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class C45:
    def fit(self, X, y):
        
        self.model = DecisionTreeClassifier(criterion='entropy')
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

class C50:
    def fit(self, X, y):
        
        self.model = RandomForestClassifier(n_estimators=50)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    c45 = C45()
    c45.fit(X_train, y_train)
    print("C4.5 score:", c45.score(X_test, y_test))

    c50 = C50()
    c50.fit(X_train, y_train)
    print("C5.0 score:", c50.score(X_test, y_test))

    cart = CART()
    cart.fit(X_train, y_train)
    print("CART score:", cart.score(X_test, y_test))

    
    dot_data = export_graphviz(cart.model, out_file=None, 
                               feature_names=iris.feature_names,  
                               class_names=iris.target_names,  
                               filled=True, rounded=True,  
                               special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("cart_tree")  
    graph.view()  
