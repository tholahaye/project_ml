# Projet 3 : Machine Learning

Dans le cadre de la formation, nous avons eu pour mission de créer une interface de machine learning via l'API Streamlit afin de traiter deux types de dataframes fournis par le formateur.
La récupération des données se fait via une base SQL, il faut donc se connecter à la base de données. Pour ce faire, nous devons récupérer les logs contenus dans le fichier .env afin de pouvoir visualiser les travaux.

Si vous voulez juste utiliser l'application, rendez-vous sur :
https://tholahaye-project-ml-main-zqgwog.streamlit.app/

Pour utiliser le projet en local:

Pour commencer, installez les bibliothèques nécessaires dans votre environnement en utilisant le terminal.
Bien qu'il soit possible de les installer directement sur l'ordinateur, il est recommandé de mettre en place un environnement virtuel (.venv). Voici les étapes à suivre dans l'invite de commande :

Naviguez vers le répertoire de votre projet en utilisant la commande cd, par exemple :
    cd "C:\Users\Johan\PycharmProjects\pythonProject\project_ml "
Une fois dans le répertoire, créez votre environnement virtuel avec la commande :
    virtualenv .venv
Assurez-vous d'être dans votre environnement virtuel. Vous verrez le nom de l'environnement dans votre terminal, par exemple :
 "(venv) PS C:\Users\Johan\PycharmProjects\pythonProject\project_ml>"
Si ce n'est pas le cas, activez l'environnement virtuel en exécutant le fichier "activate" avec la commande :
    .venv\Scripts\activate
 Installez ensuite les biblothèques répertoriées dans le requirements.txt via la commande :
    pip install requirements.txt

Dans le cas où le lancement se ferait en local utilisez la commande pour lancez Streamlit :
    streamlit run main.py


Voila ! l'API devrait être sous vos yeux.