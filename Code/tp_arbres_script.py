#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import graphviz

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import learning_curve


from sklearn import tree, datasets
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (10, 12)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

############################################################################
# Data Generation: example
############################################################################

np.random.seed(1)

n = 100
mu = [1., 1.]
sigma = [1., 1.]
rand_gauss(n, mu, sigma)


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

n1 = 50
n2 = 50
n3 = 50
mu1 = [1., 1.]
mu2 = [-1., -1.]
mu3 = [1., -1.]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
sigma3 = [0.9, 0.9]
data2 = rand_tri_gauss(n1, n2, n3, mu1, mu2, mu3, sigma1, sigma2, sigma3)

n1 = 50
n2 = 50
sigma1 = 1.
sigma2 = 5.
data3 = rand_clown(n1, n2, sigma1, sigma2)


n1 = 114  
n2 = 114
n3 = 114
n4 = 114
sigma = 0.1
data4 = rand_checkers(n1, n2, n3, n4, sigma)

#%%
############################################################################
# Displaying labeled data
############################################################################

plt.close("all")
plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(141)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(142)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(143)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.subplot(144)
plt.title('Fourth data set')
plot_2d(data4[:, :2], data4[:, 2], w=None)

#%%
############################################
# ARBRES
############################################


# Q2. Créer deux objets 'arbre de décision' en spécifiant le critère de
# classification comme l'indice de gini ou l'entropie, avec la
# fonction 'DecisionTreeClassifier' du module 'tree'.

dt_entropy = tree.DecisionTreeClassifier(criterion='entropy')
dt_gini = tree.DecisionTreeClassifier(criterion='gini')

# Effectuer la classification d'un jeu de données simulées avec rand_checkers des échantillons de
# taille n = 456 (attention à bien équilibrer les classes)

data = rand_checkers(n1=114,n2=114,n3=114,n4=114,sigma=0.1)
n_samples = len(data)
X_train = data[:, :2]
Y_train = data[:, 2].astype(int)

dt_gini.fit(X_train, Y_train)
dt_entropy.fit(X_train, Y_train)

print("Gini criterion")
print(dt_gini.get_params())
print(dt_gini.score(X_train, Y_train))

print("Entropy criterion")
print(dt_entropy.get_params())
print(dt_entropy.score(X_train, Y_train))

#%%
# Afficher les scores en fonction du paramètre max_depth

dmax = 12
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

plt.figure(figsize=(15, 10))
for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy',max_depth=i+1)
    dt_entropy.fit(X_train,Y_train)
    scores_entropy[i] = dt_entropy.score(X_train, Y_train)

    dt_gini = tree.DecisionTreeClassifier(criterion='gini',max_depth=i+1)
    dt_gini.fit(X_train,Y_train)
    scores_gini[i] = dt_gini.score(X_train, Y_train)

    plt.subplot(3, 4, i + 1)
    frontiere(lambda x: dt_gini.predict(x.reshape((1, -1))), X_train, Y_train, step=50, samples=False)
plt.draw()

plt.figure()
plt.plot(1-scores_entropy, label='entropy')
plt.plot(1-scores_gini, label='gini')
plt.xlabel('Max depth')
plt.ylabel('Accuracy Score')
plt.legend()
plt.title("Entropy and Gini Criterion Accuracy Scores in terms of Max Depth")
plt.draw()
print("Scores with entropy criterion: ", scores_entropy)
print("Scores with Gini criterion: ", scores_gini)

#%%
# Q3 Afficher la classification obtenue en utilisant la profondeur qui minimise le pourcentage d’erreurs
# obtenues avec l’entropie

dt_entropy.max_depth = 10

plt.figure()
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X_train, Y_train, step=100)
plt.title("Frontier with Minimum Error from Entropy Criterion")
plt.draw()
print("Best score with entropy criterion: ", dt_entropy.score(X_train, Y_train))

#%%
# Q4.  Exporter la représentation graphique de l'arbre: Need graphviz installed
# Voir https://scikit-learn.org/stable/modules/tree.html#classification

tree.plot_tree(dt_entropy)
dot_data = tree.export_graphviz(dt_entropy, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("arbre", format='pdf')

#%%
# Q5 :  Génération d'une base de test
data_test = rand_checkers(40, 40, 40, 40)
X_test = data_test[:, :2]
Y_test = data_test[:, 2].astype(int)

dmax = 20
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)
plt.figure(figsize=(6, 7))

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy',
                                             max_depth=i + 1)
    dt_entropy.fit(X_train, Y_train)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X_train, Y_train)
    scores_gini[i] = dt_gini.score(X_test, Y_test)

plt.draw()
plt.figure()
plt.plot(1-scores_entropy, label='entropy')
plt.plot(1-scores_gini, label='gini')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Accuracy Score')
plt.title('Testing Error')
print(scores_entropy)

#%%
# Q6. même question avec les données de reconnaissances de texte 'digits'

digits = datasets.load_digits()

X, y = digits.data, digits.target

ntest = len(digits.data)-int(len(digits.data)*0.8)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

X_train = digits.data[:ntest]
Y_train = digits.target[:ntest]
X_test = digits.data[ntest:]
Y_test = digits.target[ntest:]

dt_entropy = tree.DecisionTreeClassifier(criterion='entropy')
dt_gini = tree.DecisionTreeClassifier(criterion='gini')

dt_entropy.fit(X_train, Y_train)
dt_gini.fit(X_train, Y_train)

print(dt_entropy)
print(dt_entropy.get_params())

#%%

dmax = 15
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

plt.figure(figsize=(15, 10))
for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy',
                                             max_depth=i+1)
    dt_entropy.fit(X_train, Y_train)
    scores_entropy[i] = dt_entropy.score(X_train, Y_train)

    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X_train, Y_train)
    scores_gini[i] = dt_gini.score(X_train, Y_train)
plt.draw()


plt.figure()
plt.plot(1-scores_entropy, label='entropy')
plt.plot(1-scores_gini, label='gini')
plt.xlabel('Max Depth')
plt.ylabel('Training Error')
plt.legend(["entropy", "gini"])
plt.draw()
print("Errors with entropy criterion: ", 1-scores_entropy)
print("Errors with Gini criterion: ", 1-scores_gini)

#%%
dmax = 15
scores_entropy = np.zeros(dmax)
scores_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy',
                                             max_depth=i + 1)
    dt_entropy.fit(X_train, Y_train)
    scores_entropy[i] = dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    dt_gini.fit(X_train, Y_train)
    scores_gini[i] = dt_gini.score(X_test, Y_test)

plt.figure(figsize=(15,10))
plt.plot(1-scores_entropy, label='entropy')
plt.plot(1-scores_gini, label='gini')
plt.legend(["entropy", "gini"])
plt.xlabel('Max depth')
plt.ylabel('Testing Error')
plt.draw()
print("Error with entropy criterion: ", 1-scores_entropy)

#%%

# Q7. estimer la meilleur profondeur avec un cross_val_score

np.random.seed(56)

error_ent = []
error_gini = []
dmax = 16
X = digits.data
y = digits.target
for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion='entropy',
                                             max_depth=i + 1)
    accuracy = cross_val_score(dt_entropy, X, y)
    error_ent.append(1-accuracy.mean())
    dt_gini = tree.DecisionTreeClassifier(criterion='gini',
                                          max_depth=i + 1)
    accuracy2 = cross_val_score(dt_gini, X, y)
    error_gini.append(1-accuracy2.mean())

plt.figure(figsize=(7, 4))
plt.plot(error_ent, label="entropy")
plt.plot(error_gini, label="gini")
plt.xlabel('Depth')
plt.ylabel("Error")
plt.legend()
plt.title("Error with entropy and gini criterion")
plt.show()

print(error_ent)
print(error_gini)
best_depth = np.argmin(error_ent) + 1
print(best_depth)

# %%

#Q8. afficher la courbe d'apprentissage pour les arbres de déicisions sur le même jeu de données

X = digits.data
y = digits.target
model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_depth)

n_samples, train_scores, test_scores = learning_curve(model, X, y, cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.grid()
plt.fill_between(n_samples, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="blue")
plt.fill_between(n_samples, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1,
                 color="red")
plt.plot(n_samples, train_scores_mean, 'o-', color="blue",
         label="Training Score")
plt.plot(n_samples, test_scores_mean, 'o-', color="red",
         label="Cross Validation Score")
plt.legend()
plt.show()
# %%