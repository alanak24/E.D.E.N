//
//  MovieViewModel.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//
import SwiftUI

class MovieViewModel: ObservableObject {
    @Published var movies: [Movie] = []
    @Published var preferences: [Int: UserMoviePreference] = [:]

    func fetchMovies() {
        guard let url = URL(string: "https://api.themoviedb.org/3/movie/top_rated?api_key=513b40cf1f53d69edefdd4c32a9b72b2") else {
            return
        }

        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data = data else { return }

            do {
                let decoded = try JSONDecoder().decode(MovieResponse.self, from: data)

                DispatchQueue.main.async {
                    self.movies = Array(decoded.results.prefix(20))
                }

            } catch {
                print("Error decoding:", error)
            }

        }.resume()
    }
    
    func toggleLike(for movieId: Int) {
        if preferences[movieId] != nil {
            preferences[movieId]?.isLiked.toggle()
        } else {
            preferences[movieId] = UserMoviePreference(
                movieId: movieId,
                rating: nil,
                isLiked: true
            )
        }
    }

    func setRating(_ value: Double, for movieId: Int) {
        if preferences[movieId] != nil {
            preferences[movieId]?.rating = value
        } else {
            preferences[movieId] = UserMoviePreference(
                movieId: movieId,
                rating: value,
                isLiked: false
            )
        }
    }

    func setLike(_ value: Bool, for movieId: Int) {
        if preferences[movieId] != nil {
            preferences[movieId]?.isLiked = value
        } else {
            preferences[movieId] = UserMoviePreference(
                movieId: movieId,
                rating: nil,
                isLiked: value
            )
        }
    }

    
    
}
