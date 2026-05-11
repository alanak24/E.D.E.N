//
//  MovieDetailView.swift
//  EDEN
//
//  Created by Alana Kumar on 1/5/2026.
//

import SwiftUI

struct MovieDetailView: View {
    
    @EnvironmentObject var movieVM: MovieViewModel
    let movie: Movie
    var ratingBinding: Binding<Double> {
        Binding(
            get: {
                movieVM.preferences[movie.id]?.rating ?? 0
            },
            set: { newValue in
                movieVM.setRating(newValue, for: movie.id)
            }
        )
    }

    var isLiked: Bool {
        movieVM.preferences[movie.id]?.isLiked ?? false
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                // Poster
                AsyncImage(
                    url: URL(string:
                        "https://image.tmdb.org/t/p/w500\(movie.poster_path)"
                    )
                ) { image in
                    image
                        .resizable()
                        .scaledToFit()
                } placeholder: {
                    ProgressView()
                }

                // Title
                Text(movie.title)
                    .font(.title)
                    .foregroundColor(.white)

                // Overview
                Text(movie.overview)
                    .foregroundColor(.gray)
                
                VStack(alignment: .leading, spacing: 12) {
                    
                    Text("Your Rating")
                        .font(.headline)
                    
                    StarRatingView(rating: ratingBinding)
                    
                    Button(action: {
                        movieVM.toggleLike(for: movie.id)
                    }) {
                        HStack {
                            Image(systemName: isLiked ? "heart.fill" : "heart")
                            Text(isLiked ? "Liked" : "Add to Likes")
                        }
                        .foregroundColor(.red)
                    }
                }
                .padding(.vertical)

            }
            .padding()
        }
        .background(Color.black)
        .navigationTitle("Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}

