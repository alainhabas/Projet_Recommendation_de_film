import pandas as pd
from sklearn.neighbors import NearestNeighbors
from unidecode import unidecode
import streamlit as st

pd.set_option('display.width', 7000000)
pd.set_option('display.max_columns', 100)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++ THE APP STARTS HERE ++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
st.title('Film Recommendation App')

def hidden_features():
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )

    a = st.sidebar.radio('Select one:', [1, 2])
    if a == 1:
        st.write("hola")

#"https://www.imdb.com/title/" + title_id + "/"

# Chargement de la base principale
@st.cache
def loading_dataframe():
    # Cache la base de base
    df_full_final_X = pd.read_csv('https://media.githubusercontent.com/media/Dinoxel/film_reco_app/master/Desktop/projets/projet_2/database_imdb/df_full_final_X.csv', index_col=0)

    # Store la base d'affichage
    df_display_final_def = df_full_final_X.copy()[['titleId', 'title', 'multigenres', 'startYear', 'runtimeMinutes', 'averageRating', 'numVotes', 'nconst']]
    df_display_final_def['nconst'] = df_display_final_def['nconst'].astype(str)

    # Store la base de knn
    df_knn_final_def = df_full_final_X.copy().drop(columns=['averageRating', 'numVotes', 'startYear', 'runtimeMinutes', 'multigenres', 'years', 'nconst'])
    return df_display_final_def, df_knn_final_def

# Assignation de la DB principale aux bases d'affichage et de machine learning
df_display_final_X, df_knn_final_X = loading_dataframe()

#df_posters = pd.read_pickle(gen_link('df_posters'))

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++ WEIGHTS ++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Selectionne toutes les colonnes booléennes
X = df_knn_final_X.iloc[:, 2:].columns
# Définit le poids de base pour toutes les colonnes sur 1
df_weights = pd.DataFrame([[1 for x in X]], columns=X)

# Définit les poids de chaque partie
weight_genres = 0.65
weight_rating_low = 0.75
weight_reals = 1.25
weight_actors = 0.5
weight_years = 0.85
weight_years_low = 0.75
weight_numvotes_low = 0.5
weight_numvotes_med = 0.65

df_genres = df_display_final_X.copy()
df_genres["multigenres"] = df_genres["multigenres"].str.split(',')
df_genres = df_genres.explode("multigenres")
film_genres = df_genres["multigenres"].value_counts().head(7).index

# Gère le poids des genres
for film_genre in film_genres:
    df_weights[film_genre] = weight_genres

# Gère le poids du rating <= 7.5
df_weights['rating <= 7.5'] = weight_rating_low

# Gère le poids des réalisateurs
for real_type in df_weights.loc[:, 'nm0000229':'year <= 1960']:
    df_weights[real_type] = weight_reals

# Gère le poids des acteurs
for actor_type in df_weights.loc[:, 'year >= 1990':'nm9654612']:
    df_weights[actor_type] = weight_actors

# Gère le poids des années
for year_type in df_weights.loc[:, 'year <= 1960':'year >= 1990']:
    df_weights[year_type] = weight_years

df_weights['year <= 1960'] = weight_years_low

weights = df_weights.iloc[0].to_list()
df_weights['numvotes <= 3.6k'] = weight_numvotes_low
df_weights['3.6k < numvotes > 16k'] = weight_numvotes_med

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++ INPUT +++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Définit la base à utiliser pour la recherche
df_display_titles = df_display_final_X[['titleId', 'title', 'numVotes', 'startYear', 'multigenres']]

# Demande un film à chercher
film_title = unidecode(st.text_input('Définissez un film pour obtenir des recommendations', key="1")).lower()

# Condition si la demande fait moins de 3 lettres, repose la question
if not film_title:
    st.write("")
elif len(film_title) <= 2:
    st.warning("L'objet de la recherche doit comporter au moins 3 lettres")
else:
    is_custom_word = False
    custom_words_dict = {'lotr': 'Le Seigneur des anneaux',
                         'star wars': 'Star Wars',
                         'harry potter': 'Harry Potter',
                         'indiana jones': 'Indiana Jones'}

    # Condition pour les noms de saga, modifie 'film_title' pour la recherche
    for acronym, saga_name in custom_words_dict.items():
        if film_title == acronym:
            cleaned_name = saga_name
            film_title = unidecode(saga_name).lower()
            is_custom_word = True
            break

    # Recherche le film demandé dans la base de données
    df_display_titles = df_display_titles[df_display_titles['title'].apply(lambda x: unidecode(x.lower())).str.contains(film_title)]

    # Si au moins un film correspond à la recherche

    if not len(df_display_titles) > 0:
        st.warning("Aucun film ne correspond à votre recherche, veuillez en choisir un autre.")
    else:

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++ INPUT INDEX ++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        film_index = 123456789
        # condition si un seul film présent après recherche
        if len(df_display_titles) == 1:
            film_index = 0

        # condition si plusieurs film ont le même nom
        else:
            st.write("Les films suivants ressortent d'après votre recherche :")

            # condition pour la recherche par saga
            if is_custom_word:
                df_display_titles = df_display_titles.sort_values(by=['startYear', 'title']).reset_index()
                st.dataframe(df_display_titles[['startYear', 'title']])
                text_index_input = f"Choissisez l'index du film souhaité pour la saga '{cleaned_name}' :"

            # condition pour la recherche standard
            else:
                df_display_titles = df_display_titles.sort_values(by='numVotes', ascending=False).reset_index()
                st.dataframe(df_display_titles[['startYear', 'title', 'multigenres']])
                first_film = df_display_titles.iloc[0]
                text_index_input = "Pour le sélectionner écrivez 0, sinon écrivez le numéro du film souhaité :"
                st.write(f"Le film le plus pertinent semble être '{first_film.title}' de {first_film.startYear}.")

            selected_film = st.text_input(text_index_input, key="2")

            if selected_film:
                # condition si l'index n'est pas dans la liste
                if int(selected_film) not in list(range(len(df_display_titles))):
                    st.warning("Il semblerait que l'index", str(selected_film), "ne soit pas dans la liste, veuillez sléctionner un index valide")
                else:
                    film_index = df_display_titles.index[int(selected_film)]
            else:
                st.write("")

        # condition pour éviter au code de fonctionner si aucun paramètre n'a été rentré
        # bidouillage
        if film_index == 123456789:
            st.write('')
        else:

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++ MACHINE LEARNING ++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            st.write("Vous avez sélectionné le film : '" + df_display_titles.iloc[film_index, :].title + "'")
            film_id = df_display_titles.iloc[film_index, :].titleId
            selected_film = df_display_final_X[df_display_final_X['titleId'] == film_id].iloc[:1]

            # Définit des infos pour le ML
            X = df_knn_final_X.iloc[:, 2:]
            weights = df_weights.iloc[0].to_list()
            n_neighbors_num = max_film_length = 6

            # MACHINE LEARNING
            while True:
                model_nn = NearestNeighbors(n_neighbors=n_neighbors_num, metric_params={"w": weights}, metric='wminkowski').fit(X)
                selected_films_index = model_nn.kneighbors(df_knn_final_X[df_knn_final_X['titleId'] == film_id].iloc[:1, 2:])[1][0][1:]

                # Augmente les voisins n si le film est présent dans la liste de recommendation
                # Bidouillage
                if len([df_knn_final_X.iloc[x, 0] for x in selected_films_index if x != film_id]) != max_film_length:
                    n_neighbors_num += 1
                else:
                    break

            # Si le film est présent, le supprime des films à afficher
            selected_films_index = selected_films_index.tolist()
            if selected_film.index.to_list()[0] in selected_films_index:
                selected_films_index.remove(selected_film.index.to_list()[0])
            else:
                selected_films_index = selected_films_index[:-1]

            # Concatène les films prédits
            predicted_films = pd.DataFrame()
            for film_index in selected_films_index:
                predicted_films = pd.concat([predicted_films, df_display_final_X.iloc[[film_index]]], ignore_index=True)

            # Affiche le filmé sélectionné
            st.dataframe(selected_film)

            # Affiche la recommendation de films
            st.dataframe(predicted_films)


            # get_html_title_page('0110912')
            #print(predicted_films)

def display_files():

    LOGO_IMAGE = "https://www.themoviedb.org/t/p/original/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #f9a01b !important;
            padding-top: 75px !important;
        }
        .logo-img {
            margin-right:1%;
            float:right;
            width: 19%;
        }
        
        .container .logo-img:last-child {
        margin-right:0;
        }
        .container {
        justify-content: space-between;
        display:flex
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    a =     """<div class="container">
            <img class="logo-img" src="{LOGO_IMAGE}">
            <p class="logo-text">{name}</p>
        </div>"""


    st.markdown(f"""
        <div class="container">
        {[print(f'<img class="logo-img" src="{LOGO_IMAGE}">') for x in range(5)]}
        </div>
        """,
        unsafe_allow_html=True
    )
