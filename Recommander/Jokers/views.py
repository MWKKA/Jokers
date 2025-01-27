from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from .recommendation_engine import MovieRecommender  #importer la logique de recommandation

CSV_PATH = "Jokers/static/final_df.csv"

#initialiser le moteur de recommandation
recommender = MovieRecommender.from_csv(CSV_PATH)

def search_movie(request):
    """Page d'accueil avec le formulaire de recherche."""
    return render(request, 'search_movie.html')

def recommend_movies(request):
    """Génère des recommandations pour un film donné."""
    movie_title = request.GET.get('movie_name', '').strip()
    start_year = request.GET.get('start_year', '').strip()
    n_recommendations = request.GET.get('n_recommendations', 5)  # Valeur par défaut

    #vérif entrées utilisateur
    if not movie_title:
        return HttpResponse("Veuillez entrer un nom de film.", status=400)

    try:
        #convertir start_year si fournie
        start_year = int(start_year) if start_year else None
        n_recommendations = int(n_recommendations) if n_recommendations else 5

        #vérifier n_recommendations = positif
        if n_recommendations <= 0:
            return HttpResponse("Le nombre de recommandations doit être positif.", status=400)

        #obtenir recommandations
        recommendations = recommender.recommend(
            movie_title, 
            start_year=start_year, 
            n_recommendations=n_recommendations
        )

        #titre premier film (ou un message si aucune recommandation)
        movie_title_display = recommendations[0]['primaryTitle'] if recommendations else "Aucun film trouvé"
        
        #renvoyer la réponse avec les recommandations
        return render(request, 'recommendations.html', {
            'movie_title': movie_title,
            'start_year': start_year,
            'n_recommendations': n_recommendations,
            'recommended_movies': recommendations,
        })
    
    except ValueError as e:
        return HttpResponse(f"Erreur : {e}", status=400)

def search_movies(request):
    """Recherche des films correspondant à une entrée utilisateur."""
    query = request.GET.get('q', '').strip().lower()
    if query:
        try:
            #recherche films correspondant à l'entrée
            matching_movies = recommender.df[
                recommender.df['primaryTitle'].str.lower().str.contains(query)
            ].head(10)  #limite 10 films

            #préparer les résultats au format JSON
            results = matching_movies[['primaryTitle', 'startYear', 'poster_url']].to_dict(orient='records')
            return JsonResponse(results, safe=False)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse([], safe=False)
